import os
import subprocess
import traceback

import torch

import videoseal
from embedder import Embedder
from utils import get_random_msg

model = videoseal.load("videoseal")
message = get_random_msg()
images = torch.clamp(torch.randn(1, 1, 256, 256), 0, 1)
embedder = Embedder(model.embedder).eval()
onnx_file = "models/embedder_256b_256_256.onnx"
trt_file = "models/embedder_256b_256_256_fp16.trt"

try:
    torch.onnx.export(
        embedder,
        (images, message),
        onnx_file,
        export_params=True,
        input_names=["frame", "message"],
        output_names=["watermarked_frame"],
    )

    process = subprocess.Popen(
        ["trtexec", f"--onnx={onnx_file}", f"--saveEngine={trt_file}"]
    )
    process.wait()
    assert process.returncode == 0
except Exception:
    print(traceback.print_exc())
finally:
    if os.path.exists(onnx_file):
        os.remove(onnx_file)
