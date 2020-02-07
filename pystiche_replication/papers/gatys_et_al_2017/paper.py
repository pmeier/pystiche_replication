from typing import Union, Any, Optional, Sequence, Tuple, Dict
import logging
import torch
from torch import nn
from pystiche.enc import MultiLayerEncoder
from pystiche.ops import (
    MSEEncodingOperator,
    GramOperator,
    MultiRegionOperator,
)
from pystiche.pyramid import ImagePyramid
from pystiche_replication.utils import get_input_image, make_reproducible
from pystiche_replication.optim import default_image_pyramid_optim_loop
from .utils import (
    gatys_et_al_2017_preprocessor,
    gatys_et_al_2017_postprocessor,
    gatys_et_al_2017_multi_layer_encoder,
    gatys_et_al_2017_optimizer,
    GatysEtAl2017StyleLoss,
    GuidedGramOperator,
    GatysEtAl2017PerceptualLoss,
    GatysEtAl2017GuidedPerceptualLoss,
)

__all__ = [
    "gatys_et_al_2017_content_loss",
    "gatys_et_al_2017_style_loss",
    "gatys_et_al_2017_guided_style_loss",
    "gatys_et_al_2017_perceptual_loss",
    "gatys_et_al_2017_guided_perceptual_loss",
    "gatys_et_al_2017_image_pyramid",
    "gatys_et_al_2017_nst",
    "gatys_et_al_2017_guided_nst",
]


def gatys_et_al_2017_content_loss(
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu_4_2",
    score_weight=1e0,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()
    encoder = multi_layer_encoder[layer]

    return MSEEncodingOperator(encoder, score_weight=score_weight)


def gatys_et_al_2017_style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Optional[Union[str, Sequence[float]]] = None,
    score_weight: float = 1e3,
    **gram_op_kwargs: Any,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

    if layers is None:
        layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")

    def get_encoding_op(encoder, layer_weight):
        return GramOperator(encoder, score_weight=layer_weight, **gram_op_kwargs)

    return GatysEtAl2017StyleLoss(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        impl_params=impl_params,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def gatys_et_al_2017_guided_style_loss(
    regions: Sequence[str],
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    region_weights: Union[str, Sequence[float]] = "sum",
    layer_weights: Optional[Union[str, Sequence[float]]] = None,
    score_weight: float = 1e3,
    **gram_op_kwargs: Any,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

    if layers is None:
        layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")

    def get_encoding_op(encoder, layer_weight):
        return GuidedGramOperator(encoder, score_weight=layer_weight, **gram_op_kwargs)

    def get_region_op(region, region_weight):
        return GatysEtAl2017StyleLoss(
            multi_layer_encoder,
            layers,
            get_encoding_op,
            impl_params=impl_params,
            layer_weights=layer_weights,
            score_weight=region_weight,
        )

    return MultiRegionOperator(
        regions, get_region_op, region_weights=region_weights, score_weight=score_weight
    )


def gatys_et_al_2017_perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss = gatys_et_al_2017_content_loss(
        multi_layer_encoder=multi_layer_encoder, **content_loss_kwargs
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = gatys_et_al_2017_style_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    return GatysEtAl2017PerceptualLoss(content_loss, style_loss)


def gatys_et_al_2017_guided_perceptual_loss(
    regions: Sequence[str],
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss = gatys_et_al_2017_content_loss(
        multi_layer_encoder=multi_layer_encoder, **content_loss_kwargs
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = gatys_et_al_2017_guided_style_loss(
        regions,
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    return GatysEtAl2017GuidedPerceptualLoss(content_loss, style_loss)


def gatys_et_al_2017_image_pyramid(
    edge_sizes: Sequence[int] = (500, 800),
    num_steps: Union[int, Sequence[int]] = (500, 200),
    **image_pyramid_kwargs,
):
    return ImagePyramid(edge_sizes, num_steps, **image_pyramid_kwargs)


def gatys_et_al_2017_nst(
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
        criterion = gatys_et_al_2017_perceptual_loss(impl_params=impl_params)

    if pyramid is None:
        pyramid = gatys_et_al_2017_image_pyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_resize = pyramid[-1].resize_image
    content_image = initial_resize(content_image)
    style_image = initial_resize(style_image)
    input_image = get_input_image(starting_point="content", content_image=content_image)

    preprocessor = gatys_et_al_2017_preprocessor().to(device)
    postprocessor = gatys_et_al_2017_postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))
    criterion.set_style_image(preprocessor(style_image))

    return default_image_pyramid_optim_loop(
        input_image,
        criterion,
        pyramid,
        get_optimizer=gatys_et_al_2017_optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
    )


def gatys_et_al_2017_guided_nst(
    content_image: torch.Tensor,
    content_guides: Dict[str, torch.Tensor],
    style_images_and_guides: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    impl_params: bool = True,
    criterion: Optional[nn.Module] = None,
    pyramid: Optional[ImagePyramid] = None,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
    seed: int = None,
) -> torch.Tensor:
    regions = set(content_guides.keys())
    if regions != set(style_images_and_guides.keys()):
        # FIXME
        raise RuntimeError
    regions = sorted(regions)

    if seed is not None:
        make_reproducible()

    if criterion is None:
        criterion = gatys_et_al_2017_guided_perceptual_loss(
            regions, impl_params=impl_params
        )

    if pyramid is None:
        pyramid = gatys_et_al_2017_image_pyramid(resize_targets=(criterion,))

    device = content_image.device
    criterion = criterion.to(device)

    initial_image_resize = pyramid[-1].resize_image
    initial_guide_resize = pyramid[-1].resize_guide

    content_image = initial_image_resize(content_image)
    content_guides = {
        region: initial_guide_resize(guide) for region, guide in content_guides.items()
    }
    style_images_and_guides = {
        region: (initial_image_resize(image), initial_guide_resize(guide))
        for region, (image, guide) in style_images_and_guides.items()
    }
    input_image = get_input_image(starting_point="content", content_image=content_image)

    preprocessor = gatys_et_al_2017_preprocessor().to(device)
    postprocessor = gatys_et_al_2017_postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))

    for region, (image, guide) in style_images_and_guides.items():
        criterion.set_style_guide(region, guide)
        criterion.set_style_image(region, preprocessor(image))

    for region, guide in content_guides.items():
        criterion.set_content_guide(region, guide)

    return default_image_pyramid_optim_loop(
        input_image,
        criterion,
        pyramid,
        get_optimizer=gatys_et_al_2017_optimizer,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
    )
