import cupy as cp
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as f


class DetectorTRT:
    def __init__(self, trt_path) -> None:
        logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(logger)
        with open(trt_path, "rb") as fp:
            self.engine = self.runtime.deserialize_cuda_engine(fp.read())

        self.context = self.engine.create_execution_context()
        self.bindings = [None] * self.engine.num_io_tensors
        OUTPUT = "message"

        self.op_dtype = trt.nptype(self.engine.get_tensor_dtype(OUTPUT))
        self.op_shape = self.engine.get_tensor_shape(OUTPUT)

    def forward(self, images):
        images = cp.from_dlpack(torch.utils.dlpack.to_dlpack(images))
        message = cp.zeros(self.op_shape, self.op_dtype)

        bindings = [
            int(images.data.ptr),
            int(message.data.ptr),
        ]

        exec_status = self.context.execute_v2(bindings)
        assert exec_status

        output = torch.utils.dlpack.from_dlpack(message.toDlpack())
        return output


class FrameDetectorTRT(nn.Module):
    def __init__(self, detector_path) -> None:
        super().__init__()
        device = torch.device("cuda")
        self.det = DetectorTRT(detector_path)
        self.im_size = (256, 256)
        self.device = device

    def post_process(self, x):
        conf = x[:, 0]
        bits = x[:, 1:]
        return conf, bits

    @torch.no_grad
    def forward(self, images):
        """
        Returns confidence (b 1) and bits (b k)
        k is the size of the bits, k = 256 by default
        """
        images = images / 255.0
        images = images.permute(0, 3, 1, 2)
        images = f.interpolate(
            images,
            self.im_size,
            mode="bilinear",
            align_corners=True,
            antialias=True,
        )
        x = self.det.forward(images)
        return self.post_process(x)
