# Creates a calibrated INT8 version of the embedder

import os
import pathlib
import tempfile
import time

import cupy as cp
import cv2
import numpy as np
import tensorrt as trt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor
from tqdm import tqdm

import videoseal
from src.trt_models import Embedder, FrameEmbedderTRT
from src.utils import bit_accuracy_256b, get_message

transforms = Compose(
    [
        ToTensor(),
        Resize((256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, data_path, transform):
        super().__init__()
        path = pathlib.Path(data_path)
        self.image_list = list(path.rglob("**/*.jpg"))
        self.transform = transform

    def __getitem__(self, index):
        image_bgr_np = cv2.imread(self.image_list[index])
        image_rgb_np = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)
        return self.transform(image_rgb_np), torch.tensor(get_message().reshape(-1))

    def __len__(self):
        return len(self.image_list)


def calibration_data_loader(batch_size=8):
    dataset = ImageDataset("data/images", transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    for images, messages in loader:
        yield images.numpy(), messages.numpy()


class EmbedderEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file, image_shape, message_shape):
        super().__init__()
        self.data_loader = data_loader
        self.cache_file = cache_file
        self.image_input_shape = image_shape
        image_size = trt.volume(image_shape) * np.float32().nbytes
        image_buffer = cp.cuda.alloc(image_size)
        image_mem = cp.cuda.UnownedMemory(
            ptr=int(image_buffer), size=image_size, owner=self
        )
        mem_pointer = cp.cuda.MemoryPointer(image_mem, 0)
        self.image_inp = cp.ndarray(
            shape=image_shape, dtype=cp.float32, memptr=mem_pointer
        )

        message_size = trt.volume(message_shape) * np.float32().nbytes
        message_buffer = cp.cuda.alloc(message_size)
        message_mem = cp.cuda.UnownedMemory(
            ptr=int(message_buffer), size=message_size, owner=self
        )
        mem_pointer = cp.cuda.MemoryPointer(message_mem, 0)
        self.message_inp = cp.ndarray(
            shape=message_shape, dtype=cp.float32, memptr=mem_pointer
        )

    def get_batch_size(self):
        return self.image_input_shape[0]

    def get_batch(self, names):
        try:
            images, messages = next(self.data_loader)
            images = np.ascontiguousarray(images, dtype=np.float32)
            messages = np.ascontiguousarray(messages, dtype=np.float32)
            images_cp = cp.asarray(images)
            messages_cp = cp.asarray(messages)
        except StopIteration:
            return None

        # batch shape: (N, C, H, W)
        cp.copyto(self.image_inp, images_cp)
        cp.copyto(self.message_inp, messages_cp)

        return [int(self.image_inp.data.ptr), int(self.message_inp.data.ptr)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine(engine_path):
    temp_dir = tempfile.TemporaryDirectory()
    onnx_path = os.path.join(temp_dir.name, "embedder.onnx")
    model = videoseal.load("videoseal")
    message = torch.tensor(get_message())
    images = torch.clamp(torch.randn(1, 3, 256, 256), 0, 1)
    embedder = Embedder(model.embedder).eval()
    torch.onnx.export(
        embedder,
        (images, message),
        onnx_path,
        export_params=True,
        input_names=["frame", "message"],
        output_names=["watermarked_frame"],
    )
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)

    image_shape = (8, 3, 256, 256)
    message_shape = (8, 256)
    calibrator = EmbedderEntropyCalibrator(
        data_loader=calibration_data_loader(),
        cache_file="int8.cache",
        image_shape=image_shape,
        message_shape=message_shape,
    )
    config.int8_calibrator = calibrator
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    temp_dir.cleanup()


def benchmark(trt_file):
    batch_size = 1

    device = torch.device("cuda")
    trt_model = FrameEmbedderTRT(trt_file)

    transforms_2 = Compose(
        [
            ToTensor(),
        ]
    )
    dataset = ImageDataset("data/images", transforms_2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    detector = torch.jit.load("models/y_256b_img.jit").to(device)

    mean_accuracy = 0
    mean_time_s = 0
    with torch.no_grad():
        for idx, (images, messages) in tqdm(enumerate(loader, start=1)):
            images = images.to(device)
            messages = messages.to(device)
            st = time.perf_counter()
            watermarked_frames = trt_model(images.permute(0, 2, 3, 1), messages)
            print(watermarked_frames.shape)
            end = time.perf_counter()
            time_taken_s = end - st
            mean_time_s = mean_time_s * (idx - 1) / idx + time_taken_s / idx
            det_out = detector.detect(watermarked_frames.permute(0, 3, 1, 2) / 255.0)
            bits = det_out[:, 1:]
            watermark = torch.clamp(bits, 0, 1)
            acc = float(bit_accuracy_256b(watermark, messages).cpu())
            mean_accuracy = mean_accuracy * (idx - 1) / idx + acc / idx

    print(f"Mean accuracy: {mean_accuracy}, Mean time: {mean_time_s * 1000:.2f}[ms]")


if __name__ == "__main__":
    trt_file = "models/embedder.trt"
    build_int8_engine(trt_file)
    benchmark(trt_file)
