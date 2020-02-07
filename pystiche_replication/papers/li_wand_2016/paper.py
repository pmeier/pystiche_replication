from typing import Any, Union, Optional, Sequence, Tuple, Dict
import logging
import torch
from torch import nn
from pystiche.enc import MultiLayerEncoder
from pystiche.ops import MultiLayerEncodingOperator
from pystiche.pyramid import OctaveImagePyramid, ImagePyramid
from pystiche_replication.utils import get_input_image, make_reproducible
from pystiche_replication.optim import default_image_pyramid_optim_loop
from .utils import (
    li_wand_2016_preprocessor,
    li_wand_2016_postprocessor,
    li_wand_2016_multi_layer_encoder,
    li_wand_2016_optimizer,
    LiWand2016MSEEncodingOperator,
    LiWand2016MRFOperator,
    LiWand2016TotalVariationOperator,
    LiWand2016PerceptualLoss,
)

__all__ = [
    "li_wand_2016_content_loss",
    "li_wand_2016_style_loss",
    "li_wand_2016_regularization",
    "li_wand_2016_perceptual_loss",
    "li_wand_2016_nst",
]


def li_wand_2016_content_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu_4_2",
    score_weight: Optional[float] = None,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = li_wand_2016_multi_layer_encoder()
    encoder = multi_layer_encoder[layer]

    if score_weight is None:
        score_weight = 2e1 if impl_params else 1e0

    return LiWand2016MSEEncodingOperator(
        encoder, impl_params=impl_params, score_weight=score_weight
    )


def li_wand_2016_style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Union[str, Sequence[float]] = "sum",
    patch_size: Union[int, Tuple[int, int]] = 3,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    num_scale_steps: Optional[int] = None,
    scale_step_width: float = 5e-2,
    num_rotation_steps: Optional[int] = None,
    rotation_step_width: float = 7.5,
    score_weight: Optional[float] = None,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = li_wand_2016_multi_layer_encoder()

    if layers is None:
        layers = ("relu_3_1", "relu_4_1")

    if stride is None:
        stride = 2 if impl_params else 1

    if num_scale_steps is None:
        num_scale_steps = 1 if impl_params else 3

    if num_rotation_steps is None:
        num_rotation_steps = 1 if impl_params else 2

    def get_encoding_op(encoder, layer_weight):
        return LiWand2016MRFOperator(
            encoder,
            patch_size,
            impl_params=impl_params,
            stride=stride,
            num_scale_steps=num_scale_steps,
            scale_step_width=scale_step_width,
            num_rotation_steps=num_rotation_steps,
            rotation_step_width=rotation_step_width,
            score_weight=layer_weight,
        )

    if score_weight is None:
        score_weight = 1e-4 if impl_params else 1e0

    return MultiLayerEncodingOperator(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def li_wand_2016_regularization(
    impl_params: bool = True, exponent: float = 2.0, score_weight: float = 1e-3,
):
    return LiWand2016TotalVariationOperator(
        impl_params=impl_params, exponent=exponent, score_weight=score_weight
    )


def li_wand_2016_perceptual_loss(
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
    content_loss = li_wand_2016_content_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **content_loss_kwargs,
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = li_wand_2016_style_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    if regularization_kwargs is None:
        regularization_kwargs = {}
    regularization = li_wand_2016_regularization(
        impl_params=impl_params, **regularization_kwargs
    )

    return LiWand2016PerceptualLoss(content_loss, style_loss, regularization)


def li_wand_2016_image_pyramid(
    impl_params: bool = True,
    max_edge_size: int = 384,
    num_steps: Optional[Union[int, Sequence[int]]] = None,
    num_levels: Optional[int] = None,
    min_edge_size: int = 64,
    edge: Union[str, Sequence[str]] = "long",
    **octave_image_pyramid_kwargs: Any,
):
    if num_steps is None:
        num_steps = 100 if impl_params else 200

    if num_levels is None:
        num_levels = 3 if impl_params else None

    return OctaveImagePyramid(
        max_edge_size,
        num_steps,
        num_levels=num_levels,
        min_edge_size=min_edge_size,
        edge=edge,
        **octave_image_pyramid_kwargs,
    )


def li_wand_2016_nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
    criterion: Optional[nn.Module] = None,
    pyramid: Optional[ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
    seed: int = None,
) -> torch.Tensor:
    if seed is not None:
        make_reproducible(seed)

    if criterion is None:
        criterion = li_wand_2016_perceptual_loss(impl_params=impl_params)

    if pyramid is None:
        pyramid = li_wand_2016_image_pyramid(resize_targets=(criterion,))

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
