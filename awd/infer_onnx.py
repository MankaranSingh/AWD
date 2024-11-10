import onnxruntime
import numpy as np

class OnnxInfer:
    def __init__(self, onnx_model_path, input_name="obs", awd=False, use_gpu=False):
        self.onnx_model_path = onnx_model_path
        self.providers = ["CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, providers=self.providers
        )
        self.input_name = input_name
        self.awd = awd

        # Automatically determine input shape
        model_inputs = self.ort_session.get_inputs()
        for input in model_inputs:
            if input.name == self.input_name:
                self.input_shape = input.shape
                break
        else:
            raise ValueError(f"Input '{self.input_name}' not found in model.")

    def infer(self, inputs):
        if self.awd:
            outputs = self.ort_session.run(None, {self.input_name: [inputs]})
            return outputs[0][0]
        else:
            outputs = self.ort_session.run(
                None, {self.input_name: inputs.astype("float32")}
            )
            return outputs[0]


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    oi = OnnxInfer(args.onnx_model_path, use_gpu=args.use_gpu)

    # Adjust input shape based on the model's input requirements
    input_shape = [dim if isinstance(dim, int) else 1 for dim in oi.input_shape]
    inputs = np.random.uniform(size=input_shape).astype(np.float32)

    times = []
    warmup = 100
    for i in range(1000):
        start = time.time()
        if i > warmup:
            times.append(time.time() - start)

    print("Average time: ", sum(times) / len(times))
    print("Average fps: ", 1 / (sum(times) / len(times)))
