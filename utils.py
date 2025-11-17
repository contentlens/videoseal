import torch


def get_random_msg(size=256):
    THRESH = 0.5
    pre_msg = torch.randn(1, size, dtype=torch.float32)
    msg = torch.zeros_like(pre_msg, dtype=torch.float32)
    pre_msg = (pre_msg - pre_msg.min()) / (pre_msg.max() - pre_msg.min())
    mask = pre_msg > THRESH
    msg[mask] = 1.0
    return msg


def create_random_images(step_size=4):
    images = torch.randn(step_size, 720, 1280, 3)
    images = (images - images.min()) / (images.max() - images.min())
    return (images * 255).to(torch.uint8)
