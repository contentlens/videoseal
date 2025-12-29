import secrets

import numpy as np
import torch


def create_image_watermark_message(size_in_bits=256):
    random_bytes = secrets.token_bytes(size_in_bits // 8)
    hex_ = random_bytes.hex()
    return hex_


def watermark_message_to_tensor(message_hex):
    num = int(message_hex, 16)
    bin_string = bin(num)[2:]
    bin_string = bin_string.zfill(len(message_hex) * 4)
    msg = np.array([int(item) for item in bin_string], dtype=np.float32)
    return msg.reshape(1, -1)


def get_message():
    msg = create_image_watermark_message()
    return watermark_message_to_tensor(msg)


def bit_accuracy_256b(preds: torch.tensor, target: torch.tensor, threshold: int = 0):
    """
    Takes in two tensors or size (B x 256) and calculates the bit accuracy.
    The predictions are thresholded before calculating the bit accuracy.
    """
    preds = (preds > threshold).to(torch.float32)
    bit_mask = (preds == target).to(torch.float32)
    mask_match = torch.sum(bit_mask, dim=-1)
    return (mask_match / 256.0).mean()
