from typing import Any, Union, Optional, Sequence, Dict
import logging
import torch
from torch import nn
from pystiche.enc import MultiLayerEncoder
from pystiche.ops import GramOperator
from pystiche_replication.utils import get_input_image, make_reproducible
from pystiche_replication.optim import default_image_optim_loop
from .utils import (
    gatys_ecker_bethge_2015_preprocessor,
    gatys_ecker_bethge_2015_postprocessor,
    gatys_ecker_bethge_2015_multi_layer_encoder,
    gatys_ecker_bethge_2015_optimizer,
    GatysEckerBethge2015MSEEncodingOperator,
    GatysEckerBethge2015StyleLoss,
    GatysEckerBethge2015PerceptualLoss,
)

__all__ = [
    "gatys_ecker_bethge_2015_content_loss",
    "gatys_ecker_bethge_2015_style_loss",
    "gatys_ecker_bethge_2015_perceptual_loss",
    "gatys_ecker_bethge_2015_nst",
]


def gatys_ecker_bethge_2015_content_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layer: str = "relu_4_2",
    score_weight: float = 1e0,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder()
    encoder = multi_layer_encoder[layer]

    return GatysEckerBethge2015MSEEncodingOperator(
        encoder, impl_params=impl_params, score_weight=score_weight
    )


def gatys_ecker_bethge_2015_style_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    layers: Optional[Sequence[str]] = None,
    layer_weights: Optional[Union[str, Sequence[float]]] = None,
    score_weight: float = 1e3,
    **gram_loss_kwargs: Any,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder()

    if layers is None:
        layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")

    def get_encoding_op(encoder, layer_weight):
        return GramOperator(encoder, score_weight=layer_weight, **gram_loss_kwargs)

    return GatysEckerBethge2015StyleLoss(
        multi_layer_encoder,
        layers,
        get_encoding_op,
        impl_params=impl_params,
        layer_weights=layer_weights,
        score_weight=score_weight,
    )


def gatys_ecker_bethge_2015_perceptual_loss(
    impl_params: bool = True,
    multi_layer_encoder: Optional[MultiLayerEncoder] = None,
    content_loss_kwargs: Optional[Dict[str, Any]] = None,
    style_loss_kwargs: Optional[Dict[str, Any]] = None,
):
    if multi_layer_encoder is None:
        multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder(
            impl_params=impl_params
        )

    if content_loss_kwargs is None:
        content_loss_kwargs = {}
    content_loss = gatys_ecker_bethge_2015_content_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **content_loss_kwargs,
    )

    if style_loss_kwargs is None:
        style_loss_kwargs = {}
    style_loss = gatys_ecker_bethge_2015_style_loss(
        impl_params=impl_params,
        multi_layer_encoder=multi_layer_encoder,
        **style_loss_kwargs,
    )

    return GatysEckerBethge2015PerceptualLoss(content_loss, style_loss)


def gatys_ecker_bethge_2015_nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    num_steps: int = 1000,
    impl_params: bool = True,
    criterion: Optional[nn.Module] = None,
    quiet: bool = False,
    logger: Optional[logging.Logger] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if seed is not None:
        make_reproducible(seed)

    if criterion is None:
        criterion = gatys_ecker_bethge_2015_perceptual_loss(impl_params=impl_params)

    device = content_image.device
    criterion = criterion.to(device)

    starting_point = "content" if impl_params else "random"
    input_image = get_input_image(
        starting_point=starting_point, content_image=content_image
    )

    preprocessor = gatys_ecker_bethge_2015_preprocessor().to(device)
    postprocessor = gatys_ecker_bethge_2015_postprocessor().to(device)

    criterion.set_content_image(preprocessor(content_image))
    criterion.set_style_image(preprocessor(style_image))

    return default_image_optim_loop(
        input_image,
        criterion,
        get_optimizer=gatys_ecker_bethge_2015_optimizer,
        num_steps=num_steps,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        quiet=quiet,
        logger=logger,
    )
