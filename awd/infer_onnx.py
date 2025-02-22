import torch
import onnxruntime
import numpy as np
import argparse
import time

class OnnxInfer:
    def __init__(self, onnx_model_path, input_name="obs", use_gpu=False):
        self.onnx_model_path = onnx_model_path
        self.providers = ["CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, providers=self.providers
        )
        self.input_name = input_name

        # Automatically determine input shape
        model_inputs = self.ort_session.get_inputs()
        for input in model_inputs:
            if input.name == self.input_name:
                self.input_shape = input.shape
                break
        else:
            raise ValueError(f"Input '{self.input_name}' not found in model.")

    def infer(self, inputs):
        # Ensure inputs have the correct batch format
        if inputs.ndim == len(self.input_shape) - 1:  # Missing batch dimension
            inputs = np.expand_dims(inputs, axis=0)  # Add batch dim
        
        outputs = self.ort_session.run(None, {self.input_name: inputs})
        return outputs[0]  # Return batch output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    oi = OnnxInfer(args.onnx_model_path, use_gpu=args.use_gpu)

    # Adjust input shape based on batch size
    input_shape = [args.batch_size] + [dim if isinstance(dim, int) else 1 for dim in oi.input_shape[1:]]
    inputs = np.random.uniform(size=input_shape).astype(np.float32)

    times = []
    warmup = 100
    for i in range(1000):
        start = time.time()
        _ = oi.infer(inputs)
        if i >= warmup:
            times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"Average time per batch: {avg_time:.6f} sec")
    print(f"Average FPS: {1 / avg_time:.2f}")
