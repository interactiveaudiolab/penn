import contextlib
import functools
import os

import torch
import tqdm

import penne


###############################################################################
# Training interface
###############################################################################


def run(
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpus=None):
    """Run model training"""
    # Distributed data parallelism
    if gpus and len(gpus) > 1:
        args = (
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            gpus)
        torch.multiprocessing.spawn(
            train_ddp,
            args=args,
            nprocs=len(gpus),
            join=True)

    else:

        # Single GPU or CPU training
        train(
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            None if gpus is None else gpus[0])

    # Return path to generator checkpoint
    return penne.checkpoint.latest_path(output_directory)


###############################################################################
# Training
###############################################################################


def train(
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpu=None):
    """Train a model"""
    # Get DDP rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = None

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(penne.RANDOM_SEED)
    train_loader, valid_loader = penne.data.loaders(dataset, gpu)

    #################
    # Create models #
    #################

    model = penne.model.Model().to(device)

    ##################################################
    # Maybe setup distributed data parallelism (DDP) #
    ##################################################

    if rank is not None:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank])

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        betas=[.80, .99],
        eps=1e-9)

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = penne.checkpoint.latest_path(
        checkpoint_directory,
        '*.pt'),

    # For some reason, returning None from latest_path returns (None,)
    path = None if path == (None,) else path

    if path is not None:

        # Load model
        (
            model,
            optimizer,
            step
        ) = penne.checkpoint.load(
            path[0],
            model,
            optimizer
        )

    else:

        # Train from scratch
        step = 0

    #####################
    # Create schedulers #
    #####################

    scheduler_fn = functools.partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=penne.LEARNING_RATE_DECAY,
        last_epoch=step // len(train_loader.dataset) if step else -1)
    scheduler = scheduler_fn(optimizer)

    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Get total number of steps
    steps = penne.NUM_STEPS

    # Setup progress bar
    if not rank:
        progress = tqdm.tqdm(
            initial=step,
            total=steps,
            dynamic_ncols=True,
            desc=f'Training {penne.CONFIG}')
    while step < steps:

        model.train()
        for batch in train_loader:

            # Unpack batch
            audio, pitch, bins, voicing = batch

            with torch.cuda.amp.autocast():

                # Forward pass
                predicted = model(audio.to(device))

                # TODO - compute losses
                losses = 0.

            ######################
            # Optimize model #
            ######################

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(losses).backward()

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            ###########
            # Logging #
            ###########

            if not rank:

                if step % penne.LOG_INTERVAL == 0:

                    # Log losses
                    scalars = {
                        'loss/total': losses,
                        'learning_rate': optimizer.param_groups[0]['lr']}
                    penne.write.scalars(log_directory, step, scalars)

                ############
                # Evaluate #
                ############

                if step % penne.EVALUATION_INTERVAL == 0:

                    evaluate(
                        log_directory,
                        step,
                        generator,
                        valid_loader,
                        gpu)

                ###################
                # Save checkpoint #
                ###################

                if step and step % penne.CHECKPOINT_INTERVAL == 0:
                    penne.checkpoint.save(
                        model,
                        optimizer,
                        step,
                        output_directory / f'{step:08d}.pt')

            # Update training step count
            if step >= steps:
                break
            step += 1

            # Update progress bar
            if not rank:
                progress.update()

        # Update learning rate every epoch
        scheduler.step()

    # Close progress bar
    if not rank:
        progress.close()

    # Save final model
    penne.checkpoint.save(
        model,
        optimizer,
        step,
        output_directory / f'{step:08d}.pt')


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, valid_loader, gpu):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # TODO - evaluate
        pass

    # Prepare generator for training
    model.train()


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(rank, dataset, directory, gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(dataset, directory, gpus)


@contextlib.contextmanager
def ddp_context(rank, world_size):
    """Context manager for distributed data parallelism"""
    # Setup ddp
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank)

    try:

        # Execute user code
        yield

    finally:

        # Close ddp
        torch.distributed.destroy_process_group()
