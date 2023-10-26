import functools

import torch
import torchutil

import penn


###############################################################################
# Training
###############################################################################


@torchutil.notify.on_return('train')
def train(datasets, directory, gpu=None):
    """Train a model"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    model = penn.model.Model().to(device)

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(penn.RANDOM_SEED)
    train_loader = penn.data.loader(datasets, 'train')
    valid_loader = penn.data.loader(datasets, 'valid')

    ####################
    # Create optimizer #
    ####################

    optimizer = torch.optim.Adam(model.parameters(), lr=penn.LEARNING_RATE)

    ##############################
    # Maybe load from checkpoint #
    ##############################

    path = torchutil.checkpoint.latest_path(directory)

    if path is not None:

        # Load model
        model, optimizer, state = torchutil.checkpoint.load(
            path,
            model,
            optimizer)
        step, epoch = state['step'], state['epoch']

    else:

        # Train from scratch
        step, epoch = 0, 0

    ##############################
    # Maybe setup early stopping #
    ##############################

    if penn.EARLY_STOPPING:
        counter = penn.EARLY_STOPPING_STEPS
        best_accuracy = 0.
        stop = False

    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Setup progress bar
    progress = penn.iterator(
        range(step, penn.STEPS),
        f'Training {penn.CONFIG}',
        step,
        penn.STEPS)
    while step < penn.STEPS and (not penn.EARLY_STOPPING or not stop):

        # Seed sampler
        train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:

            # Unpack batch
            audio, bins, *_ = batch

            with torch.autocast(device.type):

                # Forward pass
                logits = model(audio.to(device))

                # Compute losses
                losses = loss(logits, bins.to(device))

            ##################
            # Optimize model #
            ##################

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(losses).backward()

            # Update weights
            scaler.step(optimizer)

            # Update gradient scaler
            scaler.update()

            ##############
            # Evaluation #
            ##############

            # Save checkpoint
            if step and step % penn.CHECKPOINT_INTERVAL == 0:
                torchutil.checkpoint.save(
                    directory / f'{step:08d}.pt',
                    model,
                    optimizer,
                    step=step,
                    epoch=epoch)

            # Evaluate
            if step % penn.LOG_INTERVAL == 0:
                evaluate_fn = functools.partial(
                    evaluate,
                    directory,
                    step,
                    model,
                    gpu)
                evaluate_fn('train', train_loader)
                valid_accuracy = evaluate_fn('valid', valid_loader)

                # Maybe stop training
                if penn.EARLY_STOPPING:
                    counter -= 1

                    # Update best validation loss
                    if valid_accuracy > best_accuracy:
                        best_accuracy = valid_accuracy
                        counter = penn.EARLY_STOPPING_STEPS

                    # Stop training
                    elif counter == 0:
                        stop = True

            # Update training step count
            if step >= penn.STEPS or (penn.EARLY_STOPPING and stop):
                break
            step += 1

            # Update progress bar
            progress.update()

        # Update epoch
        epoch += 1

    # Close progress bar
    progress.close()

    # Save final model
    torchutil.checkpoint.save(
        directory / f'{step:08d}.pt',
        model,
        optimizer,
        step=step,
        epoch=epoch)


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, model, gpu, condition, loader):
    """Perform model evaluation"""
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Setup evaluation metrics
    metrics = penn.evaluate.Metrics()

    # Prepare model for inference
    with penn.inference_context(model):

        # Unpack batch
        for i, (audio, bins, pitch, voiced, *_) in enumerate(loader):

            # Forward pass
            logits = model(audio.to(device))

            # Update metrics
            metrics.update(
                logits.to(device),
                bins.T.to(device),
                pitch.T.to(device),
                voiced.T.to(device))

            # Stop when we exceed some number of batches
            if i + 1 == penn.LOG_STEPS:
                break

    # Format results
    scalars = {
        f'{key}/{condition}': value for key, value in metrics().items()}

    # Write to tensorboard
    torchutil.tensorboard.update(directory, step, scalars=scalars)

    return scalars[f'accuracy/{condition}']


###############################################################################
# Loss function
###############################################################################


def loss(logits, bins):
    """Compute loss function"""
    # Reshape inputs
    logits = logits.permute(0, 2, 1).reshape(-1, penn.PITCH_BINS)
    bins = bins.flatten()

    # Maybe blur target
    if penn.GAUSSIAN_BLUR:

        # Cache cents values to evaluate distributions at
        if not hasattr(loss, 'cents'):
            loss.cents = penn.convert.bins_to_cents(
                torch.arange(penn.PITCH_BINS))[:, None]

        # Ensure values are on correct device (no-op if devices are the same)
        loss.cents = loss.cents.to(bins.device)

        # Create normal distributions
        distributions = torch.distributions.Normal(
            penn.convert.bins_to_cents(bins),
            25)

        # Sample normal distributions
        bins = torch.exp(distributions.log_prob(loss.cents)).permute(1, 0)

        # Normalize
        bins = bins / (bins.max(dim=1, keepdims=True).values + 1e-8)

    else:

        # One-hot encoding
        bins = torch.nn.functional.one_hot(bins, penn.PITCH_BINS).float()

    if penn.LOSS == 'binary_cross_entropy':

        # Compute binary cross-entropy loss
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            bins)

    elif penn.LOSS == 'categorical_cross_entropy':

        # Compute categorical cross-entropy loss
        return torch.nn.functional.cross_entropy(logits, bins)

    else:

        raise ValueError(f'Loss {penn.LOSS} is not implemented')
