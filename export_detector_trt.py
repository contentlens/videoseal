import os
import subprocess
import traceback

import torch

import videoseal

model = videoseal.load("videoseal")
images = torch.randn(4, 3, 256, 256)
images = (images - images.min()) / (images.max() - images.min())
onnx_file = "models/detector_256b_256_256.onnx"
trt_file = "models/detector_256b_256_256_fp16.trt"

try:
    torch.onnx.export(
        model.detector,
        (images,),
        onnx_file,
        export_params=True,
        input_names=["frame"],
        output_names=["message"],
    )

    process = subprocess.Popen(
        ["trtexec", f"--onnx={onnx_file}", f"--saveEngine={trt_file}", "--fp16"]
    )
    process.wait()
    assert process.returncode == 0
except Exception:
    print(traceback.print_exc())
finally:
    if os.path.exists(onnx_file):
        os.remove(onnx_file)
