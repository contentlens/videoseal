import cupy as cp
import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as f

from src.attenuation import JND11


class AdditiveBlending(nn.Module):
    def __init__(self, image_scaling=1.0, pred_scaling=1.0) -> None:
        super().__init__()
        self.image_scaling = image_scaling
        self.pred_scaling = pred_scaling

    def forward(self, images, preds):
        out = self.image_scaling * images + self.pred_scaling * preds
        return out


class RGB2YUV(nn.Module):
    def __init__(self):
        super(RGB2YUV, self).__init__()
        self.register_buffer(
            "M",
            torch.tensor(
                [
                    [0.299, 0.587, 0.114],
                    [-0.14713, -0.28886, 0.436],
                    [0.615, -0.51499, -0.10001],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()  # b h w c
        yuv = x @ self.M.T
        yuv = yuv.permute(0, 3, 1, 2).contiguous()
        return yuv


class Embedder(nn.Module):
    def __init__(
        self,
        video_seal_embedder,
    ):
        super().__init__()
        self.embeddor = video_seal_embedder
        self.rgb_to_yuv = RGB2YUV()

    def forward(self, rgb, message):
        # rgb -> b c h w
        yuv = self.rgb_to_yuv(rgb)
        y = yuv[:, 0:1, ...]
        # y -> b 1 h w
        pre_watermark = self.embeddor(y, message)
        return pre_watermark


class EmbedderTRT:
    def __init__(self, trt_path) -> None:
        logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(logger)
        with open(trt_path, "rb") as fp:
            self.engine = self.runtime.deserialize_cuda_engine(fp.read())

        assert self.engine is not None, "Deserialization error"

        self.context = self.engine.create_execution_context()
        self.bindings = [None] * self.engine.num_io_tensors
        self.INP_1, self.INP_2 = "frame", "message"
        OUTPUT = "watermarked_frame"

        self.op_dtype = trt.nptype(self.engine.get_tensor_dtype(OUTPUT))
        self.op_shape = self.engine.get_tensor_shape(OUTPUT)
        self.output = cp.zeros(self.op_shape, self.op_dtype)

    def forward(self, images, message):
        images_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(images))
        message_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(message))

        bindings = [
            int(images_cp.data.ptr),
            int(message_cp.data.ptr),
            int(self.output.data.ptr),
        ]

        exec_status = self.context.execute_v2(bindings)
        assert exec_status

        output = torch.utils.dlpack.from_dlpack(cp.from_dlpack(self.output))
        return output


@torch.compile(fullgraph=True, mode="max-autotune")
def interpolate(image, height, width):
    res_first_image = f.interpolate(
        image,
        (height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return res_first_image


class FrameEmbedderTRT(nn.Module):
    def __init__(self, embedder_path, step_size) -> None:
        super().__init__()
        device = torch.device("cuda")
        self.device = device
        self.embedder = EmbedderTRT(embedder_path)
        self.attenuation = JND11().to(device)
        self.blender = AdditiveBlending(1.0, 0.2).to(device)
        self.rgb_to_yuv = RGB2YUV().to(device)
        self.im_size = (256, 256)
        self.step_size = step_size

    @torch.no_grad
    def forward(self, images, message):
        images = images.permute(0, 3, 1, 2)  # b c h w
        b, _, H, W = images.shape
        # take first image and pass through embedder
        # attenuate output to get final mask
        # interleave the final mask and additive blend across batch
        first_images = images[:: self.step_size, ...]
        res_first_images = f.interpolate(
            first_images,
            self.im_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pre_watermark_small = self.embedder.forward(res_first_images, message)
        pre_watermark = f.interpolate(
            pre_watermark_small,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        ###### [START] TODO: JIT Compile these or do something else?
        watermark = self.attenuation(first_images, pre_watermark)
        watermark_interleaved = torch.repeat_interleave(watermark, self.step_size, 0)
        result = self.blender(images, watermark_interleaved)
        result = (torch.clamp(result, 0, 1) * 255).to(torch.uint8)
        result = result.permute(0, 2, 3, 1)
        ###### [END]
        return result
