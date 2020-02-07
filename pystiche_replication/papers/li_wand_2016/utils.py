from typing import Any, Union, Optional, Sequence, Tuple, Dict, Callable
from collections import OrderedDict
import logging
import torch
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.nn.functional import mse_loss
import pystiche
from pystiche.image.transforms import CaffePreprocessing, CaffePostprocessing
from pystiche.enc import MultiLayerEncoder, Encoder, vgg19_encoder
from pystiche.ops import (
    MSEEncodingOperator,
    MRFOperator,
    TotalVariationOperator,
    MultiLayerEncodingOperator,
)
from pystiche.functional import patch_matching_loss, total_variation_loss
from pystiche.loss import MultiOperatorLoss
from pystiche.pyramid import OctaveImagePyramid
from pystiche_replication.utils import get_input_image, make_reproducible
from pystiche_replication.optim import default_image_pyramid_optim_loop

__all__ = [
    "li_wand_2016_preprocessor",
    "li_wand_2016_postprocessor",
    "li_wand_2016_multi_layer_encoder",
    "li_wand_2016_optimizer",
    "LiWand2016MSEEncodingOperator",
    "LiWand2016MRFOperator",
    "LiWand2016TotalVariationOperator",
    "LiWand2016PerceptualLoss",
]


def li_wand_2016_preprocessor() -> Callable[[torch.Tensor], torch.Tensor]:
    return CaffePreprocessing()


def li_wand_2016_postprocessor() -> Callable[[torch.Tensor], torch.Tensor]:
    return CaffePostprocessing()


def li_wand_2016_multi_layer_encoder() -> MultiLayerEncoder:
    return vgg19_encoder(weights="caffe", preprocessing=False, allow_inplace=True)


def li_wand_2016_optimizer(input_image: torch.Tensor) -> Optimizer:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


class LiWand2016MSEEncodingOperator(MSEEncodingOperator):
    def __init__(
        self, encoder: Encoder, impl_params: bool = True, **mse_encoding_op_kwargs,
    ):
        super().__init__(encoder, **mse_encoding_op_kwargs)

        self.loss_reduction = "mean" if impl_params else "sum"

    def calculate_score(self, input_repr, target_repr, ctx):
        return mse_loss(input_repr, target_repr, reduction=self.loss_reduction)


class NormalizeUnfoldGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, size, step):
        ctx.needs_normalizing = step < size
        if ctx.needs_normalizing:
            normalizer = torch.zeros_like(input)
            item = [slice(None) for _ in range(input.dim())]
            for idx in range(0, normalizer.size()[dim] - size, step):
                item[dim] = slice(idx, idx + size)
                normalizer[item].add_(1.0)

            # clamping to avoid zero division
            ctx.save_for_backward(torch.clamp(normalizer, min=1.0))
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.needs_normalizing:
            (normalizer,) = ctx.saved_tensors
            grad_input = grad_output / normalizer
        else:
            grad_input = grad_output.clone()
        return grad_input, None, None, None


normalize_unfold_grad = NormalizeUnfoldGrad.apply


def extract_normalized_patches2d(
    input: torch.Tensor, patch_size: Tuple[int, int], stride: Tuple[int, int]
) -> torch.Tensor:
    for dim, size, step in zip(range(2, input.dim()), patch_size, stride):
        input = normalize_unfold_grad(input, dim, size, step)
    return pystiche.extract_patches2d(input, patch_size, stride)


class LiWand2016MRFOperator(MRFOperator):
    def __init__(
        self,
        encoder: Encoder,
        patch_size: Union[int, Tuple[int, int]],
        impl_params: bool = True,
        **mrf_op_kwargs: Any,
    ):

        super().__init__(encoder, patch_size, **mrf_op_kwargs)

        self.normalize_patches_grad = impl_params
        self.loss_reduction = "sum"
        self.score_correction_factor = 1.0 / 2.0 if impl_params else 1.0

    def enc_to_repr(self, enc: torch.Tensor) -> torch.Tensor:
        if self.normalize_patches_grad:
            return extract_normalized_patches2d(enc, self.patch_size, self.stride)
        else:
            return pystiche.extract_patches2d(enc, self.patch_size, self.stride)

    def calculate_score(self, input_repr, target_repr, ctx):
        score = patch_matching_loss(
            input_repr, target_repr, reduction=self.loss_reduction
        )
        return score * self.score_correction_factor


class LiWand2016TotalVariationOperator(TotalVariationOperator):
    def __init__(self, impl_params: bool = True, **total_variation_op_kwargs):
        super().__init__(**total_variation_op_kwargs)

        self.loss_reduction = "sum"
        self.score_correction_factor = 1.0 / 2.0 if impl_params else 1.0

    def calculate_score(self, input_repr):
        score = total_variation_loss(
            input_repr, exponent=self.exponent, reduction=self.loss_reduction
        )
        return score * self.score_correction_factor


class LiWand2016PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
        content_loss: LiWand2016MSEEncodingOperator,
        style_loss: MultiLayerEncodingOperator,
        regularization: LiWand2016TotalVariationOperator,
    ):
        super().__init__(
            OrderedDict(
                [
                    ("content_loss", content_loss),
                    ("style_loss", style_loss),
                    ("regularization", regularization),
                ]
            )
        )

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)
