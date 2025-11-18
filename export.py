import torch

import videoseal
from embedder import Embedder
from utils import get_random_msg

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

video_seal_model = videoseal.load("videoseal")
emb = Embedder(video_seal_model.embedder.eval()).eval().half().to(device)
images = torch.randn(4, 3, 640, 640).half().to(device)
images = (images - images.min()) / (images.max() - images.min())
message = get_random_msg().half().to(device)
emb_traced = torch.jit.trace(emb, (images, message))
emb_optimized = torch.jit.optimize_for_inference(emb_traced)
emb_optimized.save("embedder_traced_256b.jit")


# det = video_seal_model.detector.to(device)
# det_traced = torch.jit.trace(det, (images,))
# det_traced.save("detector_traced_256b.jit")
