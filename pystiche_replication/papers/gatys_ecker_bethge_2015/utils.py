from typing import Union, Optional, Sequence, Callable
from collections import OrderedDict
import torch
from torch import optim, nn
from torch.optim.optimizer import Optimizer
from torch.nn.functional import mse_loss
import pystiche
from pystiche.image.transforms import CaffePreprocessing, CaffePostprocessing
from pystiche.enc import MultiLayerEncoder, Encoder, vgg19_encoder
from pystiche.ops import (
    EncodingOperator,
    MSEEncodingOperator,
    MultiLayerEncodingOperator,
)
from pystiche.loss import MultiOperatorLoss

__all__ = [
    "gatys_ecker_bethge_2015_preprocessor",
    "gatys_ecker_bethge_2015_postprocessor",
    "gatys_ecker_bethge_2015_optimizer",
    "gatys_ecker_bethge_2015_multi_layer_encoder",
    "GatysEckerBethge2015MSEEncodingOperator",
    "GatysEckerBethge2015StyleLoss",
    "GatysEckerBethge2015PerceptualLoss",
]


def gatys_ecker_bethge_2015_preprocessor() -> nn.Module:
    return CaffePreprocessing()


def gatys_ecker_bethge_2015_postprocessor() -> nn.Module:
    return CaffePostprocessing()


def gatys_ecker_bethge_2015_multi_layer_encoder(
    impl_params: bool = True,
) -> MultiLayerEncoder:
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
        score = mse_loss(input_repr, target_repr, reduction=self.loss_reduction)
        return score * self.score_correction_factor


class GatysEckerBethge2015StyleLoss(MultiLayerEncodingOperator):
    def __init__(
        self,
        layers: Sequence[str],
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        multi_layer_encoder: MultiLayerEncoder,
        impl_params: bool = True,
        layer_weights: Optional[Union[str, Sequence[float]]] = None,
        score_weight: float = 1e0,
    ):
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
        score = super().process_input_image(input_image)
        return score * self.score_correction_factor


class GatysEckerBethge2015PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
        content_loss: GatysEckerBethge2015MSEEncodingOperator,
        style_loss: GatysEckerBethge2015StyleLoss,
    ):
        super().__init__(
            OrderedDict([("content_loss", content_loss), ("style_loss", style_loss)])
        )

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)
