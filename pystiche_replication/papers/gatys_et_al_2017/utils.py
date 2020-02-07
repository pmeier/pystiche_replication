from typing import Union, Optional, Sequence, Callable
from collections import OrderedDict
import torch
from torch import optim
from torch.optim.optimizer import Optimizer
import pystiche
from pystiche.image.transforms import CaffePreprocessing, CaffePostprocessing
from pystiche.enc import MultiLayerEncoder, Encoder, vgg19_encoder
from pystiche.ops import (
    EncodingOperator,
    EncodingComparisonGuidance,
    MSEEncodingOperator,
    GramOperator,
    MultiLayerEncodingOperator,
)
from pystiche.loss import MultiOperatorLoss


__all__ = [
    "gatys_et_al_2017_preprocessor",
    "gatys_et_al_2017_postprocessor",
    "gatys_et_al_2017_multi_layer_encoder",
    "gatys_et_al_2017_optimizer",
    "GatysEtAl2017StyleLoss",
    "GuidedGramOperator",
    "GatysEtAl2017PerceptualLoss",
    "GatysEtAl2017GuidedPerceptualLoss",
]


def gatys_et_al_2017_preprocessor() -> Callable[[torch.Tensor], torch.Tensor]:
    return CaffePreprocessing()


def gatys_et_al_2017_postprocessor() -> Callable[[torch.Tensor], torch.Tensor]:
    return CaffePostprocessing()


def gatys_et_al_2017_multi_layer_encoder() -> MultiLayerEncoder:
    return vgg19_encoder(weights="caffe", preprocessing=False, allow_inplace=True)


def gatys_et_al_2017_optimizer(input_image: torch.Tensor) -> Optimizer:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


class GatysEtAl2017StyleLoss(MultiLayerEncodingOperator):
    def __init__(
        self,
        multi_layer_encoder: MultiLayerEncoder,
        layers: Sequence[str],
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        impl_params: bool = True,
        layer_weights: Optional[Union[str, Sequence[float]]] = None,
        score_weight: float = 1e0,
    ) -> None:
        if layer_weights is None:
            layer_weights = self.get_default_layer_weights(multi_layer_encoder, layers)

        super().__init__(
            multi_layer_encoder,
            layers,
            get_encoding_op,
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


class GuidedGramOperator(EncodingComparisonGuidance, GramOperator):
    pass


class _GatysEtAl2017PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self, content_loss: MSEEncodingOperator, style_loss: GatysEtAl2017StyleLoss,
    ):
        super().__init__(
            OrderedDict([("content_loss", content_loss), ("style_loss", style_loss)])
        )

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)


class GatysEtAl2017PerceptualLoss(_GatysEtAl2017PerceptualLoss):
    def __init__(
        self, content_loss: MSEEncodingOperator, style_loss: GatysEtAl2017StyleLoss,
    ):
        super().__init__(content_loss, style_loss)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)


class GatysEtAl2017GuidedPerceptualLoss(_GatysEtAl2017PerceptualLoss):
    def __init__(
        self, content_loss: MSEEncodingOperator, style_loss: GatysEtAl2017StyleLoss,
    ):

        super().__init__(content_loss, style_loss)

    def set_style_guide(self, region: str, guide: torch.Tensor):
        self.style_loss.set_target_guide(region, guide)

    def set_style_image(self, region: str, image: torch.Tensor):
        self.style_loss.set_target_image(region, image)

    def set_content_guide(self, region: str, guide: torch.Tensor):
        self.style_loss.set_input_guide(region, guide)
