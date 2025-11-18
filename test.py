import time

import torch

from detector import FrameDetectorTRT
from embedder import FrameEmbedderTRT
from utils import get_random_msg

device = torch.device("cuda")

images = torch.randn(8, 640, 640, 3).half().to(device)
images = (images - images.min()) / (images.max() - images.min())
message = get_random_msg().half().to(device)

emb = FrameEmbedderTRT("embedder_256b_256_256_fp16.trt")
det = FrameDetectorTRT("detector_256b_256_256_fp16.trt")
out = emb(images, message)
out = det(images)

start = time.perf_counter()
out = emb(images, message)
tt = time.perf_counter() - start
print(round(tt * 1000, 2), "[ms]", sep="")


start = time.perf_counter()
out = det(images)
tt = time.perf_counter() - start
print(round(tt * 1000, 2), "[ms]", sep="")
