import torch

import penne


def export(model, file):
    """Save an ONNX model from a PyTorch model"""
    # Automatic mixed precision and evaluation mode
    with penne.inference_context():

        # Model input
        audio = torch.empty(1, 1, penne.NUM_TRAINING_SAMPLES, requires_grad=True)

        # Export ONNX model
        torch.onnx.export(
            model,
            audio,
            file,
            input_names=['audio'],
            output_names=['logits'],
            dynamic_axes={
                'audio': {0: 'batch_size'},
                'logits': {0: 'batch_size'}})

        # Check validity
        if not valid(file):
            raise ValueError('Invalid ONNX model file')


def valid(file):
    """Check whether an ONNX model is valid"""
    return torch.onnx.checker.check_model(torch.onnx.load(file))
