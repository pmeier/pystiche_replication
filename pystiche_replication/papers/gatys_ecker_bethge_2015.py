from typing import Any, Union, Optional, Sequence, Dict, Callable
from collections import OrderedDict
import torch
from torch import optim, nn
from torch.optim.optimizer import Optimizer
from torch.nn.functional import mse_loss
import pystiche
from pystiche.image.transforms import CaffePreprocessing, CaffePostprocessing
from pystiche.enc import MultiLayerEncoder, Encoder, vgg19_encoder
from pystiche.ops import MSEEncodingOperator, GramOperator, MultiLayerEncodingOperator
from pystiche.loss import MultiOperatorLoss
from ..utils import get_input_image, make_reproducible
from ..optim import default_image_optim_loop

__all__ = [
    "gatys_ecker_bethge_2015_preprocessor",
    "gatys_ecker_bethge_2015_postprocessor",
    "gatys_ecker_bethge_2015_optimizer",
    "GatysEckerBethge2015MSEEncodingOperator",
    "GatysEckerBethge2015ContentLoss",
    "GatysEckerBethge2015GramOperator",
    "GatysEckerBethge2015StyleLoss",
    "GatysEckerBethge2015PerceptualLoss",
    "gatys_ecker_bethge_2015_nst",
]


def gatys_ecker_bethge_2015_preprocessor() -> Callable[[torch.Tensor], torch.Tensor]:
    return CaffePreprocessing()


def gatys_ecker_bethge_2015_postprocessor() -> Callable[[torch.Tensor], torch.Tensor]:
    return CaffePostprocessing()


def gatys_ecker_bethge_2015_multi_layer_encoder(impl_params=True) -> MultiLayerEncoder:
    multi_layer_encoder = vgg19_encoder(
        weights="caffe", preprocessing=False, allow_inplace=True
    )
    if impl_params:
        return multi_layer_encoder

    for name, module in multi_layer_encoder.named_children():
        if isinstance(module, nn.MaxPool2d):
            multi_layer_encoder._modules[name] = nn.AvgPool2d(
                **pystiche.pool_module_meta(module)
            )
    return multi_layer_encoder


def gatys_ecker_bethge_2015_optimizer(input_image: torch.Tensor) -> Optimizer:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


class GatysEckerBethge2015MSEEncodingOperator(MSEEncodingOperator):
    def __init__(
        self, encoder: Encoder, impl_params: bool = True, score_weight: float = 1e0
    ):
        super().__init__(encoder, score_weight=score_weight)

        self.score_correction_factor = 1.0 if impl_params else 1.0 / 2.0
        self.loss_reduction = "mean" if impl_params else "sum"

    def calculate_score(self, input_repr, target_repr, ctx):
        loss = mse_loss(input_repr, target_repr, reduction=self.loss_reduction)
        return loss * self.score_correction_factor


class GatysEckerBethge2015ContentLoss(GatysEckerBethge2015MSEEncodingOperator):
    def __init__(
        self,
        impl_params: bool = True,
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        layer: str = "relu_4_2",
        **mse_encoding_loss_kwargs: Any,
    ):
        if multi_layer_encoder is None:
            multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder()
        encoder = multi_layer_encoder[layer]

        super().__init__(encoder, impl_params=impl_params, **mse_encoding_loss_kwargs)


class GatysEckerBethge2015GramOperator(GramOperator):
    def __init__(self, encoder: Encoder, normalize=True, score_weight: float = 1e3):
        super().__init__(encoder, normalize=normalize, score_weight=score_weight)


class GatysEckerBethge2015StyleLoss(MultiLayerEncodingOperator):
    def __init__(
        self,
        impl_params: bool = True,
        layers: Optional[Sequence[str]] = None,
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        layer_weights: Optional[Union[str, Sequence[float]]] = None,
        score_weight: float = 1e3,
        **gram_loss_kwargs: Any,
    ):
        def get_encoding_op(encoder, layer_weight):
            return GatysEckerBethge2015GramOperator(
                encoder, score_weight=layer_weight, **gram_loss_kwargs
            )

        if multi_layer_encoder is None:
            multi_layer_encoder = gatys_ecker_bethge_2015_multi_layer_encoder()

        if layers is None:
            layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")

        if layer_weights is None:
            if impl_params:
                layer_weights = self.get_default_layer_weights(
                    multi_layer_encoder, layers
                )
            else:
                layer_weights = "mean"

        super().__init__(
            layers,
            get_encoding_op,
            multi_layer_encoder,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )

        self.score_correction_factor = 1.0 if impl_params else 1.0 / 4.0

    @staticmethod
    def get_default_layer_weights(
        multi_layer_encoder: MultiLayerEncoder, layers: Sequence[str]
    ) -> Sequence[float]:
        nums_channels = []
        for layer in layers:
            module = multi_layer_encoder._modules[layer.replace("relu", "conv")]
            nums_channels.append(module.out_channels)
        return [1.0 / num_channels ** 2.0 for num_channels in nums_channels]

    def process_input_image(self, input_image: torch.Tensor) -> pystiche.LossDict:
        return super().process_input_image(input_image) * self.score_correction_factor


class GatysEckerBethge2015PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
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
        content_loss = GatysEckerBethge2015ContentLoss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            **content_loss_kwargs,
        )

        if style_loss_kwargs is None:
            style_loss_kwargs = {}
        style_loss = GatysEckerBethge2015StyleLoss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            **style_loss_kwargs,
        )

        super().__init__(
            OrderedDict([("content_loss", content_loss), ("style_loss", style_loss)])
        )

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)


def gatys_ecker_bethge_2015_nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    num_steps: int = 1000,
    impl_params: bool = True,
    criterion: Optional[GatysEckerBethge2015PerceptualLoss] = None,
    quiet: bool = False,
    print_fn: Optional[Callable[[int, torch.Tensor], None]] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if seed is not None:
        make_reproducible(seed)

    if criterion is None:
        criterion = GatysEckerBethge2015PerceptualLoss(impl_params=impl_params)

    device = content_image.device
    criterion = criterion.to(device)

    input_image = get_input_image(starting_point="random", content_image=content_image)

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
        print_fn=print_fn,
    )
