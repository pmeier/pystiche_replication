from typing import Union, Optional
import os
import torch


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


def print_sep_line():
    print("-" * 80)


def print_replication_info(title: str, url: str, author: str, year: Union[str, int]):
    info = (
        "This is the results of the paper",
        f"'{title}'",
        url,
        "authored by",
        author,
        f"in {str(year)}",
    )
    print("\n".join(info))
    print_sep_line()
