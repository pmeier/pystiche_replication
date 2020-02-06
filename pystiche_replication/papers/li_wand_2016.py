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
    "LiWand2016ContentLoss",
    "LiWand2016MRFOperator",
    "LiWand2016StyleLoss",
    "LiWand2016TotalVariationOperator",
    "LiWand2016Regularization",
    "LiWand2016PerceptualLoss",
    "LiWand2016ImagePyramid",
    "li_wand_2016_nst",
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
        self,
        encoder: Encoder,
        impl_params: bool = True,
        score_weight: Optional[float] = None,
    ):
        if score_weight is None:
            score_weight = 2e1 if impl_params else 1e0

        super().__init__(encoder, score_weight=score_weight)

        self.loss_reduction = "mean" if impl_params else "sum"

    def calculate_score(self, input_repr, target_repr, ctx):
        return mse_loss(input_repr, target_repr, reduction=self.loss_reduction)


class LiWand2016ContentLoss(LiWand2016MSEEncodingOperator):
    def __init__(
        self,
        impl_params: bool = True,
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        layer: str = "relu_4_2",
        **mse_encoding_loss_kwargs: Any,
    ):
        if multi_layer_encoder is None:
            multi_layer_encoder = li_wand_2016_multi_layer_encoder()
        encoder = multi_layer_encoder[layer]

        super().__init__(encoder, impl_params=impl_params, **mse_encoding_loss_kwargs)


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
            normalizer, = ctx.saved_tensors
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
        impl_params: bool = True,
        patch_size: Union[int, Tuple[int, int]] = 3,
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        num_scale_steps: Optional[int] = None,
        scale_step_width: float = 5e-2,
        num_rotation_steps: Optional[int] = None,
        rotation_step_width: float = 7.5,
        score_weight: Optional[float] = None,
    ):
        if stride is None:
            stride = 2 if impl_params else 1

        if num_scale_steps is None:
            num_scale_steps = 1 if impl_params else 3

        if num_rotation_steps is None:
            num_rotation_steps = 1 if impl_params else 2

        if score_weight is None:
            score_weight = 1e-4 if impl_params else 1e0

        super().__init__(
            encoder,
            patch_size,
            stride=stride,
            num_scale_steps=num_scale_steps,
            scale_step_width=scale_step_width,
            num_rotation_steps=num_rotation_steps,
            rotation_step_width=rotation_step_width,
            score_weight=score_weight,
        )

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


class LiWand2016StyleLoss(MultiLayerEncodingOperator):
    def __init__(
        self,
        impl_params: bool = True,
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        layers: Optional[Sequence[str]] = None,
        layer_weights: Union[str, Sequence[float]] = "sum",
        score_weight: Optional[float] = None,
        **mrf_loss_kwargs: Any,
    ):
        def get_encoding_op(encoder, layer_weight):
            return LiWand2016MRFOperator(
                encoder,
                impl_params=impl_params,
                score_weight=layer_weight,
                **mrf_loss_kwargs,
            )

        if multi_layer_encoder is None:
            multi_layer_encoder = li_wand_2016_multi_layer_encoder()

        if layers is None:
            layers = ("relu_3_1", "relu_4_1")

        if score_weight is None:
            score_weight = 1e-4 if impl_params else 1e0

        super().__init__(
            layers,
            get_encoding_op,
            multi_layer_encoder,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )


class LiWand2016TotalVariationOperator(TotalVariationOperator):
    def __init__(
        self,
        impl_params: bool = True,
        exponent: float = 2.0,
        score_weight: float = 1e-3,
    ):
        super().__init__(exponent=exponent, score_weight=score_weight)

        self.loss_reduction = "sum"
        self.score_correction_factor = 1.0 / 2.0 if impl_params else 1.0

    def calculate_score(self, input_repr):
        score = total_variation_loss(
            input_repr, exponent=self.exponent, reduction=self.loss_reduction
        )
        return score * self.score_correction_factor


class LiWand2016Regularization(LiWand2016TotalVariationOperator):
    pass


class LiWand2016PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
        impl_params: bool = True,
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        content_loss_kwargs: Optional[Dict[str, Any]] = None,
        style_loss_kwargs: Optional[Dict[str, Any]] = None,
        regularization_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if multi_layer_encoder is None:
            multi_layer_encoder = li_wand_2016_multi_layer_encoder()

        if content_loss_kwargs is None:
            content_loss_kwargs = {}
        content_loss = LiWand2016ContentLoss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            **content_loss_kwargs,
        )

        if style_loss_kwargs is None:
            style_loss_kwargs = {}
        style_loss = LiWand2016StyleLoss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            **style_loss_kwargs,
        )

        if regularization_kwargs is None:
            regularization_kwargs = {}
        regularization = LiWand2016Regularization(
            impl_params=impl_params, **regularization_kwargs
        )

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


class LiWand2016ImagePyramid(OctaveImagePyramid):
    def __init__(
        self,
        impl_params: bool = True,
        max_edge_size: int = 384,
        num_steps: Optional[Union[int, Sequence[int]]] = None,
        num_levels: Optional[int] = None,
        min_edge_size: int = 64,
        edge: Union[str, Sequence[str]] = "long",
        interpolation_mode: str = "bilinear",
        resize_targets=None,
    ):
        if num_steps is None:
            num_steps = 100 if impl_params else 200

        if num_levels is None:
            num_levels = 3 if impl_params else None

        super().__init__(
            max_edge_size,
            num_steps,
            num_levels=num_levels,
            min_edge_size=min_edge_size,
            edge=edge,
            interpolation_mode=interpolation_mode,
            resize_targets=resize_targets,
        )


def li_wand_2016_nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
    criterion: Optional[LiWand2016PerceptualLoss] = None,
    pyramid: Optional[LiWand2016ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
    seed: int = None,
) -> torch.Tensor:
    if seed is not None:
        make_reproducible(seed)

    if criterion is None:
        criterion = LiWand2016PerceptualLoss(impl_params=impl_params)

    if pyramid is None:
        pyramid = LiWand2016ImagePyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_resize = pyramid[-1].resize_image
    content_image = initial_resize(content_image)
    style_image = initial_resize(style_image)
    input_image = get_input_image(starting_point="content", content_image=content_image)

    preprocessor = li_wand_2016_preprocessor().to(device)
    postprocessor = li_wand_2016_postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))
    criterion.set_style_image(preprocessor(style_image))

    return default_image_pyramid_optim_loop(
        input_image,
        criterion,
        pyramid,
        get_optimizer=li_wand_2016_optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
    )
