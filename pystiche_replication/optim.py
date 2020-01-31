from typing import Any, Union, Optional, Iterable, Callable
import torch
from torch import optim
from torch.optim.optimizer import Optimizer
from pystiche.image import extract_aspect_ratio
from pystiche.loss import LossDict
from pystiche.pyramid import ImagePyramid


def default_image_optimizer(input_image: torch.Tensor) -> Optimizer:
    return optim.LBFGS([input_image.requires_grad_(True)], lr=1.0, max_iter=1)


def default_image_optim_print_fn(
    step: int, loss: Union[torch.Tensor, LossDict]
) -> None:
    if (step + 1) % 50 == 0:
        print(loss)
        print("-" * 80)


def default_image_optim_loop(
    input_image: torch.Tensor,
    criterion: Callable[[torch.Tensor], Union[torch.Tensor, LossDict]],
    get_optimizer: Optional[Callable[[torch.Tensor], Optimizer]] = None,
    num_steps: Union[int, Iterable[int]] = 500,
    preprocessor: Callable[[torch.Tensor], torch.Tensor] = None,
    postprocessor: Callable[[torch.Tensor], torch.Tensor] = None,
    quiet: bool = False,
    print_fn: Optional[Callable[[int, Union[torch.Tensor, LossDict]], None]] = None,
) -> torch.Tensor:
    if get_optimizer is None:
        get_optimizer = default_image_optimizer

    if isinstance(num_steps, int):
        num_steps = range(num_steps)

    if print_fn is None:
        print_fn = default_image_optim_print_fn

    if preprocessor:
        with torch.no_grad():
            input_image = preprocessor(input_image)

    optimizer = get_optimizer(input_image)

    for step in num_steps:

        def closure():
            optimizer.zero_grad()

            loss = criterion(input_image)
            loss.backward()

            if not quiet:
                print_fn(step, loss)

            return float(loss)

        optimizer.step(closure)

    if postprocessor:
        with torch.no_grad():
            input_image = postprocessor(input_image)

    return input_image.detach()


def default_image_pyramid_optim_loop(
    input_image: torch.Tensor,
    criterion: Callable[[torch.Tensor], Union[torch.Tensor, LossDict]],
    pyramid: ImagePyramid,
    **image_optim_kwargs: Any,
) -> torch.Tensor:
    aspect_ratio = extract_aspect_ratio(input_image)

    for level in pyramid:
        with torch.no_grad():
            input_image = level.resize_image(input_image, aspect_ratio=aspect_ratio)

        input_image = default_image_optim_loop(
            input_image, criterion, **image_optim_kwargs
        )

    return input_image
