from typing import Union, Optional
import os
import contextlib
import torch
from pystiche_replication.optim.log import (
    PysticheLogger,
    default_logger as get_default_logger,
)

__all__ = [
    "get_images_root",
    "parse_device",
    "get_default_logger",
    "log_replication_info",
]


def get_images_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.sep.join(list(here.split(os.sep)[:-1]) + ["images"])


def parse_device(device: Optional[Union[torch.device, str]]) -> torch.device:
    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        return torch.device(device)
    elif device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ValueError("device should be a torch.device, a str, or None")


@contextlib.contextmanager
def log_replication_info(
    logger: PysticheLogger, title: str, url: str, author: str, year: Union[str, int]
):
    header = "\n".join(
        (
            "This is the replication of the paper",
            f"'{title}'",
            url,
            "authored by",
            author,
            f"in {str(year)}",
        )
    )
    with logger.environ(header):
        yield
