from typing import Union, Optional
import os
import random
import numpy as np
import torch
from pystiche.misc import verify_str_arg
from pystiche.image import read_image, write_image
from pystiche.image.transforms import GrayscaleToBinary

__all__ = [
    "read_image",
    "write_image",
    "make_reproducible",
    "get_input_image",
    "read_guides",
]


def make_reproducible(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_input_image(
    starting_point: Union[str, torch.Tensor] = "content",
    content_image: Optional[torch.tensor] = None,
    style_image: Optional[torch.tensor] = None,
):
    if isinstance(starting_point, torch.Tensor):
        return starting_point

    starting_point = verify_str_arg(
        starting_point, "starting_point", ("content", "style", "random")
    )

    if starting_point == "content":
        if content_image is not None:
            return content_image.clone()
        raise RuntimeError("starting_point is 'content', but no content image is given")
    elif starting_point == "style":
        if style_image is not None:
            return style_image.clone()
        raise RuntimeError("starting_point is 'style', but no style image is given")
    elif starting_point == "random":
        if content_image is not None:
            return torch.rand_like(content_image)
        elif style_image is not None:
            return torch.rand_like(style_image)
        raise RuntimeError("starting_point is 'random', but no image is given")


def read_guides(root, image_file, device):
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    guide_folder = os.path.join(root, image_name)
    transform = GrayscaleToBinary()
    return {
        os.path.splitext(guide_file)[0]: transform(
            read_image(os.path.join(guide_folder, guide_file)).to(device)
        )
        for guide_file in os.listdir(guide_folder)
    }
