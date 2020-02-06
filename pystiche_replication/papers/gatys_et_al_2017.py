from typing import Union, Any, Optional, Sequence, Tuple, Dict, Callable
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
    MultiRegionOperator,
)
from pystiche.loss import MultiOperatorLoss
from pystiche.pyramid import ImagePyramid
from ..utils import get_input_image, make_reproducible
from ..optim import default_image_pyramid_optim_loop

__all__ = [
    "gatys_et_al_2017_preprocessor",
    "gatys_et_al_2017_postprocessor",
    "gatys_et_al_2017_multi_layer_encoder",
    "gatys_et_al_2017_optimizer",
    "GatysEtAl2017MSEEncodingOperator",
    "GatysEtAl2017ContentLoss",
    "GatysEtAl2017GramOperator",
    "GatysEtAl2017GuidedGramOperator",
    "GatysEtAl2017StyleLoss",
    "GatysEtAl2017GuidedStyleLoss",
    "GatysEtAl2017PerceptualLoss",
    "GatysEtAl2017GuidedPerceptualLoss",
    "GatysEtAl2017ImagePyramid",
    "gatys_et_al_2017_nst",
    "gatys_et_al_2017_guided_nst",
]


def gatys_et_al_2017_preprocessor() -> Callable[[torch.Tensor], torch.Tensor]:
    return CaffePreprocessing()


def gatys_et_al_2017_postprocessor() -> Callable[[torch.Tensor], torch.Tensor]:
    return CaffePostprocessing()


def gatys_et_al_2017_multi_layer_encoder() -> MultiLayerEncoder:
    return vgg19_encoder(weights="caffe", preprocessing=False, allow_inplace=True)


def gatys_et_al_2017_optimizer(input_image: torch.Tensor) -> Optimizer:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


class GatysEtAl2017MSEEncodingOperator(MSEEncodingOperator):
    pass


class GatysEtAl2017ContentLoss(GatysEtAl2017MSEEncodingOperator):
    def __init__(
        self,
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        layer: str = "relu_4_2",
        **mse_encoding_loss_kwargs: Any,
    ):
        if multi_layer_encoder is None:
            multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()
        encoder = multi_layer_encoder[layer]

        super().__init__(encoder, **mse_encoding_loss_kwargs)


class GatysEtAl2017GramOperator(GramOperator):
    def __init__(
        self, encoder: Encoder, normalize: bool = True, score_weight: float = 1e3
    ):
        super().__init__(encoder, normalize=normalize, score_weight=score_weight)


class GatysEtAl2017GuidedGramOperator(
    EncodingComparisonGuidance, GatysEtAl2017GramOperator
):
    pass


class _GatysEtAl2017StyleLoss(MultiLayerEncodingOperator):
    def __init__(
        self,
        get_encoding_op: Callable[[Encoder, float], EncodingOperator],
        impl_params: bool = True,
        layers: Optional[Sequence[str]] = None,
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        layer_weights: Optional[Union[str, Sequence[float]]] = None,
        score_weight: float = 1e3,
    ) -> None:
        if multi_layer_encoder is None:
            multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

        if layers is None:
            layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")

        if layer_weights is None:
            layer_weights = self.get_default_layer_weights(multi_layer_encoder, layers)

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


class GatysEtAl2017StyleLoss(_GatysEtAl2017StyleLoss):
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
            return GatysEtAl2017GramOperator(
                encoder, score_weight=layer_weight, **gram_loss_kwargs
            )

        super().__init__(
            get_encoding_op,
            impl_params=impl_params,
            layers=layers,
            multi_layer_encoder=multi_layer_encoder,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )


class GatysEtAl2017RegionStyleLoss(_GatysEtAl2017StyleLoss):
    def __init__(
        self,
        impl_params: bool = True,
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        layers: Optional[Sequence[str]] = None,
        layer_weights: Optional[Union[str, Sequence[float]]] = None,
        score_weight: float = 1e3,
        **gram_loss_kwargs: Any,
    ):
        def get_encoding_op(encoder, layer_weight):
            return GatysEtAl2017GuidedGramOperator(
                encoder, score_weight=layer_weight, **gram_loss_kwargs
            )

        super().__init__(
            get_encoding_op,
            impl_params=impl_params,
            layers=layers,
            multi_layer_encoder=multi_layer_encoder,
            layer_weights=layer_weights,
            score_weight=score_weight,
        )


class GatysEtAl2017GuidedStyleLoss(MultiRegionOperator):
    def __init__(
        self,
        regions: Sequence[str],
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        region_weights: Union[str, Sequence[float]] = "sum",
        score_weight: float = 1e3,
        **style_loss_kwargs: Any,
    ):
        if multi_layer_encoder is None:
            multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

        def get_op(region, region_weight):
            return GatysEtAl2017RegionStyleLoss(
                multi_layer_encoder=multi_layer_encoder,
                score_weight=region_weight,
                **style_loss_kwargs,
            )

        super().__init__(
            regions, get_op, region_weights=region_weights, score_weight=score_weight
        )


class _GatysEtAl2017PerceptualLoss(MultiOperatorLoss):
    def __init__(
        self,
        content_loss: GatysEtAl2017ContentLoss,
        style_loss: _GatysEtAl2017StyleLoss,
    ):
        super().__init__(
            OrderedDict([("content_loss", content_loss), ("style_loss", style_loss)])
        )

    def set_content_image(self, image: torch.Tensor):
        self.content_loss.set_target_image(image)


class GatysEtAl2017PerceptualLoss(_GatysEtAl2017PerceptualLoss):
    def __init__(
        self,
        impl_params: bool = True,
        multi_layer_encoder: Optional[MultiLayerEncoder] = None,
        content_loss_kwargs: Optional[Dict[str, Any]] = None,
        style_loss_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if multi_layer_encoder is None:
            multi_layer_encoder = gatys_et_al_2017_multi_layer_encoder()

        if content_loss_kwargs is None:
            content_loss_kwargs = {}
        content_loss = GatysEtAl2017ContentLoss(
            multi_layer_encoder=multi_layer_encoder, **content_loss_kwargs
        )

        if style_loss_kwargs is None:
            style_loss_kwargs = {}
        style_loss = GatysEtAl2017StyleLoss(
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            **style_loss_kwargs,
        )

        super().__init__(content_loss, style_loss)

    def set_style_image(self, image: torch.Tensor):
        self.style_loss.set_target_image(image)


class GatysEtAl2017GuidedPerceptualLoss(_GatysEtAl2017PerceptualLoss):
    def __init__(
        self,
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
        content_loss = GatysEtAl2017ContentLoss(
            multi_layer_encoder=multi_layer_encoder, **content_loss_kwargs
        )

        if style_loss_kwargs is None:
            style_loss_kwargs = {}
        style_loss = GatysEtAl2017GuidedStyleLoss(
            regions,
            impl_params=impl_params,
            multi_layer_encoder=multi_layer_encoder,
            **style_loss_kwargs,
        )

        super().__init__(content_loss, style_loss)

    def set_style_guide(self, region: str, guide: torch.Tensor):
        self.style_loss.set_target_guide(region, guide)

    def set_style_image(self, region: str, image: torch.Tensor):
        self.style_loss.set_target_image(region, image)

    def set_content_guide(self, region: str, guide: torch.Tensor):
        self.style_loss.set_input_guide(region, guide)


class GatysEtAl2017ImagePyramid(ImagePyramid):
    def __init__(
        self,
        edge_sizes: Sequence[int] = (500, 700),
        num_steps: Union[int, Sequence[int]] = (500, 200),
        edge: Union[str, Sequence[str]] = "short",
        interpolation_mode: str = "bilinear",
        resize_targets=None,
    ):
        super().__init__(
            edge_sizes,
            num_steps,
            edge=edge,
            interpolation_mode=interpolation_mode,
            resize_targets=resize_targets,
        )


def gatys_et_al_2017_nst(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    impl_params: bool = True,
    criterion: Optional[GatysEtAl2017PerceptualLoss] = None,
    pyramid: Optional[GatysEtAl2017ImagePyramid] = None,
    quiet: bool = False,
    print_fn: Optional[Callable[[int, torch.Tensor], None]] = None,
    seed: int = None,
) -> torch.Tensor:
    if seed is not None:
        make_reproducible(seed)

    if criterion is None:
        criterion = GatysEtAl2017PerceptualLoss(impl_params=impl_params)

    if pyramid is None:
        pyramid = GatysEtAl2017ImagePyramid(resize_targets=(criterion,))

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
        print_fn=print_fn,
    )


def gatys_et_al_2017_guided_nst(
    content_image: torch.Tensor,
    content_guides: Dict[str, torch.Tensor],
    style_images_and_guides: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    impl_params: bool = True,
    criterion: Optional[GatysEtAl2017GuidedPerceptualLoss] = None,
    pyramid: Optional[GatysEtAl2017ImagePyramid] = None,
    quiet: bool = False,
    print_fn: Optional[Callable[[int, torch.Tensor], None]] = None,
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
        criterion = GatysEtAl2017GuidedPerceptualLoss(regions, impl_params=impl_params)

    if pyramid is None:
        pyramid = GatysEtAl2017ImagePyramid(resize_targets=(criterion,))

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
        print_fn=print_fn,
    )
