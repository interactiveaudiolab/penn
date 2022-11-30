import onnxruntime
import torch

import penne


def export(model, file):
    """Save an ONNX model from a PyTorch model"""
    model = model.cpu()

    # Evaluation mode
    with penne.inference_context(model):

        # Model input
        audio = torch.empty(
            1,
            1,
            penne.NUM_TRAINING_SAMPLES,
            requires_grad=True)

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
        valid(file)


def model(file):
    """Initialize an ONNX model for inference"""
    return onnxruntime.InferenceSession(file)


def valid(file):
    """Check whether an ONNX model is valid"""
    import onnx
    onnx.checker.check_model(onnx.load(file))
