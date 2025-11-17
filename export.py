import torch

import videoseal
from embedder import Embedder, FrameEmbedder
from utils import get_random_msg

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

video_seal_model = videoseal.load("videoseal")
emb = Embedder(video_seal_model.embedder).to(device)
images = torch.randn(1, 3, 256, 256).to(device)
images = (images - images.min()) / (images.max() - images.min())
message = get_random_msg().to(device)
emb_traced = torch.jit.trace(emb, (images, message))
emb_traced.save("embedder_traced_256b.jit")
emb = FrameEmbedder("embedder_traced_256b.jit", device=device)
out = emb(images.permute(0, 2, 3, 1), message)
print(out.device)


det = video_seal_model.detector.to(device)
det_traced = torch.jit.trace(det, (images,))
det_traced.save("detector_traced_256b.jit")
