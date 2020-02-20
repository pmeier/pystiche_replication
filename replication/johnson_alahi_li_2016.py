import os
from os import path
from argparse import Namespace
import torch
from pystiche.image import write_image
from pystiche_replication import (
    johnson_alahi_li_2016_training,
    johnson_alahi_li_2016_images,
    johnson_alahi_li_2016_stylization,
)
from pystiche_replication.optim import OptimLogger
from pystiche_replication.papers.johnson_alahi_li_2016.data import (
    johnson_alahi_li_2016_dataset,
    johnson_alahi_li_2016_image_loader,
)
from pystiche_replication.papers.common_utils import save_state_dict


def training(args):
    content = "chicago"
    styles = ("starry_night", "la_muse", "composition_vii", "the_wave")

    dataset = johnson_alahi_li_2016_dataset(
        args.dataset_dir, impl_params=args.impl_params
    )
    image_loader = johnson_alahi_li_2016_image_loader(
        dataset, pin_memory=str(args.device).startswith("cuda"),
    )

    images = johnson_alahi_li_2016_images(root=args.image_dir)
    content_image = images[content].read(device=args.device)

    for style in styles:
        style_image = images[style].read(device=args.device)

        transformer = johnson_alahi_li_2016_training(
            image_loader,
            style_image,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
            quiet=args.quiet,
            logger=args.logger,
        )
        save_state_dict(transformer, style)

        output_image = johnson_alahi_li_2016_stylization(
            content_image,
            transformer,
            impl_params=args.impl_params,
            instance_norm=args.instance_norm,
        )
        output_file = f"{content}__{style}.jpg"
        write_image(output_image, output_file)


def parse_input():
    # TODO: write CLI
    image_dir = None
    # dataset_dir = None
    dataset_dir = "/home/philipmeier/hdd_raid/coco/train2014"
    model_dir = None
    device = None
    impl_params = True
    instance_norm = False
    quiet = False

    def process_dir(dir):
        dir = path.abspath(path.expanduser(dir))
        os.makedirs(dir, exist_ok=True)
        return dir

    here = path.dirname(__file__)

    if image_dir is None:
        image_dir = path.join(here, "data", "images", "source")
    image_dir = process_dir(image_dir)

    if dataset_dir is None:
        dataset_dir = path.join(here, "data", "images", "dataset")
    dataset_dir = process_dir(dataset_dir)

    if model_dir is None:
        model_dir = path.join(here, "data", "models")
    model_dir = process_dir(model_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    logger = OptimLogger()

    return Namespace(
        image_dir=image_dir,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        device=device,
        impl_params=impl_params,
        instance_norm=instance_norm,
        logger=logger,
        quiet=quiet,
    )


if __name__ == "__main__":
    args = parse_input()

    training(args)
