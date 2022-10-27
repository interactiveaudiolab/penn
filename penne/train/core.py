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
    datasets,
    checkpoint_directory,
    output_directory,
    log_directory,
    gpus=None):
    """Run model training"""
    # Distributed data parallelism
    if gpus and len(gpus) > 1:
        args = (
            datasets,
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
            datasets,
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
    datasets,
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
    train_loader = penne.data.loader(datasets, 'train', gpu)
    valid_loader = penne.data.loader(datasets, 'valid', gpu)

    ################
    # Create model #
    ################

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

    optimizer = torch.optim.Adam(model.parameters(), lr=penne.LEARNING_RATE)

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = penne.checkpoint.latest_path(checkpoint_directory, '*.pt')

    if path is not None:

        # Load model
        model, optimizer, step = penne.checkpoint.load(path, model, optimizer)

    else:

        # Train from scratch
        step = 0

    ##########################################
    # Maybe setup adaptive gradient clipping #
    ##########################################

    if penne.ADAPTIVE_CLIPPING:

        # Don't apply gradient clipping to the linear layer of CREPE
        parameters = [
            list(module.parameters()) for name, module in model.named_modules()
            if name != 'classifier']

        #
        optimizer = penne.train.clip.AdaptiveGradientClipping(
            parameters,
            optimizer)

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
            audio, bins, *_ = batch

            with torch.cuda.amp.autocast():

                # Forward pass
                logits = model(audio.to(device))

                # Compute losses
                losses = loss(logits, bins.to(device))

            ######################
            # Optimize model #
            ######################

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(losses).backward()

            # Maybe unscale for gradient clipping
            if penne.ADAPTIVE_CLIPPING:
                scaler.unscale_(optimizer)

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            ###########
            # Logging #
            ###########

            if not rank:

                ############
                # Evaluate #
                ############

                if step % penne.EVALUATION_INTERVAL == 0:
                    evaluate_fn = functools.partial(
                        evaluate,
                        log_directory,
                        step,
                        model,
                        gpu)
                    evaluate_fn('train', train_loader)
                    evaluate_fn('valid', valid_loader)

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


def evaluate(directory, step, model, gpu, condition, loader):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Prepare model for evaluation
    model.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Setup evaluation metrics
        metrics = penne.evaluate.Metrics()

        for i, batch in enumerate(loader):

            # Unpack batch
            audio, bins, pitch, voiced, *_ = batch

            # Forward pass
            logits = model(audio.to(device))

            # Update metrics
            metrics.update(
                logits,
                bins.to(device),
                pitch.to(device),
                voiced.to(device))

            # Stop when we exceed some number of batches
            if i + 1 == penne.EVALUATION_STEPS:
                break

        # Format results
        scalars = {
            f'{key}/{condition}': value for key, value in metrics().items()}

        # Write to tensorboard
        penne.write.scalars(directory, step, scalars)

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


###############################################################################
# Loss function
###############################################################################


def loss(logits, bins):
    """Compute loss function"""
    # Reshape inputs
    logits = logits.permute(0, 2, 1).reshape(-1, penne.PITCH_BINS)
    bins = bins.flatten()

    # Maybe blur target
    if penne.GAUSSIAN_BLUR:

        # Cache cents values to evaluate distributions at
        if not hasattr(loss, 'cents'):
            loss.cents = penne.convert.bins_to_cents(
                torch.arange(penne.PITCH_BINS))[:, None]

        # Ensure values are on correct device (no-op if devices are the same)
        loss.cents = loss.cents.to(bins.device)

        # Create normal distributions
        distributions = torch.distributions.Normal(
            penne.convert.bins_to_cents(bins),
            25)

        # Sample normal distributions
        bins = torch.exp(distributions.log_prob(loss.cents)).permute(1, 0)

        # Normalize
        bins = bins / (bins.max(dim=1, keepdims=True).values + 1e-8)
    else:

        # One-hot encoding
        bins = torch.nn.functional.one_hot(bins, penne.PITCH_BINS).float()

    if penne.LOSS == 'binary_cross_entropy':

        # Positive example weight
        weight = torch.full(
            (penne.PITCH_BINS,),
            penne.BCE_POSITIVE_WEIGHT,
            dtype=torch.float,
            device=logits.device)

        # Compute binary cross-entropy loss
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            bins,
            pos_weight=weight)

    elif penne.LOSS == 'categorical_cross_entropy':

        # Compute categorical cross-entropy loss
        return torch.nn.functional.cross_entropy(logits, bins)

    else:

        raise ValueError(f'Loss {penne.LOSS} is not implemented')
