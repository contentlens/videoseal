import torch
import torch.nn as nn


class JND11(nn.Module):
    def __init__(self) -> None:
        """
        in_channels = 1
        out_channls = 1
        """
        super(JND11, self).__init__()

        # setup input and output methods
        self.in_channels = 1
        groups = self.in_channels

        # create kernels
        kernel_x = (
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_y = (
            torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_lum = (
            torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 2.0, 2.0, 2.0, 1.0],
                    [1.0, 2.0, 0.0, 2.0, 1.0],
                    [1.0, 2.0, 2.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # Expand kernels for 3 input channels and 3 output channels, apply the same filter to each channel
        kernel_x = kernel_x.repeat(groups, 1, 1, 1)
        kernel_y = kernel_y.repeat(groups, 1, 1, 1)
        kernel_lum = kernel_lum.repeat(groups, 1, 1, 1)

        self.conv_x = nn.Conv2d(
            3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups
        )
        self.conv_y = nn.Conv2d(
            3, 3, kernel_size=(3, 3), padding=1, bias=False, groups=groups
        )
        self.conv_lum = nn.Conv2d(
            3, 3, kernel_size=(5, 5), padding=2, bias=False, groups=groups
        )

        self.conv_x.weight = nn.Parameter(kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y, requires_grad=False)
        self.conv_lum.weight = nn.Parameter(kernel_lum, requires_grad=False)

        # image values to be (0, 1)
        self.register_buffer(
            "rgb_to_gray_mat",
            torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1) * 255,
        )

        # setup apply mode

    def jnd_la(self, x, alpha=1.0, eps=1e-5):
        """Luminance masking: x must be in [0,255]"""
        la = self.conv_lum(x) / 32
        mask_lum = la <= 127
        la[mask_lum] = 17 * (1 - torch.sqrt(la[mask_lum] / 127 + eps))
        la[~mask_lum] = 3 / 128 * (la[~mask_lum] - 127) + 3
        return alpha * la

    def jnd_cm(self, x, beta=0.117, eps=1e-5):
        """Contrast masking: x must be in [0,255]"""
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        cm = torch.sqrt(grad_x**2 + grad_y**2)
        cm = 16 * cm**2.4 / (cm**2 + 26**2)
        return beta * cm

    def heatmaps(self, imgs: torch.Tensor, clc: float = 0.3) -> torch.Tensor:
        """imgs must be in [0,1]"""
        mat = imgs * self.rgb_to_gray_mat  # type: ignore
        gray = torch.sum(mat, dim=1)
        la = self.jnd_la(gray)
        cm = self.jnd_cm(gray)
        hmaps = torch.clamp_min(
            la + cm - clc * torch.minimum(la, cm), 0
        )  # b 1 h w
        return hmaps / 255

    def forward(
        self, images: torch.Tensor, watermark: torch.Tensor
    ) -> torch.Tensor:
        """imgs and deltas must be in [0,1]"""
        hmaps = self.heatmaps(images)
        watermark = hmaps * watermark
        return watermark
