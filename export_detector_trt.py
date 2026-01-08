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
from src.detector_trt import FrameDetectorTRT
from src.utils import bit_accuracy_256b, get_message

BATCH_SIZE = 4


class ImageDatasetDetector(Dataset):
    def __init__(self, data_path, videoseal_jit_path, transform, out_transform=None):
        super().__init__()
        path = pathlib.Path(data_path)
        self.image_list = list(path.rglob("**/*.jpg"))
        self.transform = transform
        self.out_transform = out_transform
        self.model = torch.jit.load(videoseal_jit_path)

    def __getitem__(self, index):
        image_bgr_np = cv2.imread(self.image_list[index])
        image_rgb_np = cv2.cvtColor(image_bgr_np, cv2.COLOR_BGR2RGB)
        image = self.transform(image_rgb_np)
        message = torch.tensor(get_message().reshape(-1))
        with torch.no_grad():
            out = self.model.embed(
                image.unsqueeze(0), message.reshape(1, -1)
            ).cpu()  # b c h w, float32, [0.0, 1.0]
        if self.out_transform:
            out = self.out_transform(out).squeeze()
        else:
            out = out.squeeze()
        return out, message

    def __len__(self):
        return len(self.image_list)


def calibration_data_loader(batch_size=8):
    out_transform = Compose(
        [
            Resize(
                (256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
        ]
    )
    transform = Compose(
        [
            ToTensor(),
            Resize(
                (256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
        ]
    )
    dataset = ImageDatasetDetector(
        "data/images", "models/y_256b_img.jit", transform, out_transform=out_transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    for images, _ in loader:
        yield images.numpy()


class EmbedderEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file, image_shape):
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

    def get_batch_size(self):
        return self.image_input_shape[0]

    def get_batch(self, names):
        try:
            images = next(self.data_loader)
            images = np.ascontiguousarray(images, dtype=np.float32)
            images_cp = cp.asarray(images)
        except StopIteration:
            return None

        # batch shape: (N, C, H, W)
        cp.copyto(self.image_inp, images_cp)

        return [int(self.image_inp.data.ptr)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine(engine_path):
    image_size = (BATCH_SIZE, 3, 256, 256)

    temp_dir = tempfile.TemporaryDirectory()
    onnx_path = os.path.join(temp_dir.name, "detector.onnx")
    model = videoseal.load("videoseal")
    images = torch.clamp(torch.randn(*image_size), 0, 1)
    detector = model.detector.eval()
    torch.onnx.export(
        detector,
        (images,),
        onnx_path,
        export_params=True,
        input_names=["frame"],
        output_names=["message"],
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

    calibrator = EmbedderEntropyCalibrator(
        data_loader=calibration_data_loader(BATCH_SIZE),
        cache_file="detector_int8.cache",
        image_shape=image_size,
    )
    config.int8_calibrator = calibrator
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)
    config.builder_optimization_level = 0

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    temp_dir.cleanup()


def build_fp16_engine(engine_path):
    image_size = (BATCH_SIZE, 3, 256, 256)

    temp_dir = tempfile.TemporaryDirectory()
    onnx_path = os.path.join(temp_dir.name, "detector.onnx")
    model = videoseal.load("videoseal")
    images = torch.clamp(torch.randn(*image_size), 0, 1)
    detector = model.detector.eval()
    torch.onnx.export(
        detector,
        (images,),
        onnx_path,
        export_params=True,
        input_names=["frame"],
        output_names=["message"],
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
    config.builder_optimization_level = 0

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    temp_dir.cleanup()


def benchmark(trt_file):
    batch_size = 4

    device = torch.device("cuda")
    trt_model = FrameDetectorTRT(trt_file)

    transform = Compose(
        [
            ToTensor(),
            Resize(
                (1080, 1080), interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
        ]
    )

    out_transforms = Compose(
        [
            Resize(
                (1080, 1080), interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
        ]
    )

    dataset = ImageDatasetDetector(
        "data/images", "models/y_256b_img.jit", transform, out_transforms
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    mean_accuracy = 0
    mean_time_s = 0
    with torch.no_grad():
        for idx, (images, messages) in tqdm(enumerate(loader, start=1)):
            images = images.to(device)
            messages = messages.to(device)
            st = time.perf_counter()
            _, bits = trt_model(images.permute(0, 2, 3, 1))
            end = time.perf_counter()
            time_taken_s = end - st
            mean_time_s = mean_time_s * (idx - 1) / idx + time_taken_s / idx
            watermark = torch.clamp(bits, 0, 1)
            acc = float(bit_accuracy_256b(watermark, messages).cpu())

            mean_accuracy = mean_accuracy * (idx - 1) / idx + acc / idx

    print(f"Mean accuracy: {mean_accuracy}, Mean time: {mean_time_s * 1000:.2f}[ms]")


if __name__ == "__main__":
    trt_file = "models/detector.trt"
    build_int8_engine(trt_file)
    # build_fp16_engine(trt_file)
    benchmark(trt_file)
