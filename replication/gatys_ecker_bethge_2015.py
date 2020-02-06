from os import path
import itertools
from pystiche.cuda import abort_if_cuda_memory_exausts
from pystiche_replication import (
    GatysEckerBethge2015PerceptualLoss,
    gatys_ecker_bethge_2015_nst,
)
from pystiche_replication.utils import read_image as _read_image, write_image
import utils


# guessed
NUM_STEPS = 1000
SIZE = 800


def read_image(file, **kwargs):
    return _read_image(file, size=SIZE, **kwargs)


@abort_if_cuda_memory_exausts
def figure_2(source_folder, replication_folder, device, impl_params, logger, quiet):
    content_file = path.join(source_folder, "praefcke__tuebingen_neckarfront.jpg")
    content_image = read_image(content_file, device=device)

    class StyleImage:
        def __init__(self, label, file, score_weight):
            self.label = label
            self.image = read_image(path.join(source_folder, file), device=device)
            self.score_weight = score_weight

    style_images = (
        StyleImage("B", "turner__shipwreck_of_the_minotaur.jpg", 1e3),
        StyleImage("C", "van_gogh__starry_night.jpg", 1e3),
        StyleImage("D", "munch__the_scream.jpg", 1e3),
        StyleImage("E", "picasso__figure_dans_un_fauteuil.jpg", 1e4),
        StyleImage("F", "kandinsky__composition_vii.jpg", 1e4),
    )

    params = "implementation" if impl_params else "paper"
    for style_image in style_images:
        header = f"Replicating Figure 2 {style_image.label} with {params} parameters"
        with logger.environ(header):

            style_loss_kwargs = {"score_weight": style_image.score_weight}
            criterion = GatysEckerBethge2015PerceptualLoss(
                impl_params=impl_params, style_loss_kwargs=style_loss_kwargs
            )

            output_image = gatys_ecker_bethge_2015_nst(
                content_image,
                style_image.image,
                NUM_STEPS,
                impl_params=impl_params,
                criterion=criterion,
                quiet=quiet,
                logger=logger,
            )

            output_file = path.join(
                replication_folder, "fig_2__{}.jpg".format(style_image.label)
            )
            logger.sep_message(f"Saving result to {output_file}", bottom=False)
            write_image(output_image, output_file)


@abort_if_cuda_memory_exausts
def figure_3(source_folder, results_folder, device, impl_params, logger, quiet):
    content_file = path.join(source_folder, "praefcke__tuebingen_neckarfront.jpg")
    content_image = read_image(content_file, device=device)

    style_file = path.join(source_folder, "kandinsky__composition_vii.jpg")
    style_image = read_image(style_file, device=device)

    style_layers = ("relu_1_1", "relu_2_1", "relu_3_1", "relu_4_1", "relu_5_1")
    layer_configs = [style_layers[: idx + 1] for idx in range(len(style_layers))]

    score_weights = (1e5, 1e4, 1e3, 1e2)

    for layers, score_weight in itertools.product(layer_configs, score_weights):
        row_label = layers[-1].replace("relu_", "Conv")
        column_label = f"{1.0 / score_weight:.0e}"
        header = f"Replicating Figure 3 row {row_label} and column {column_label}"
        with logger.environ(header):

            style_loss_kwargs = {"layers": layers, "score_weight": score_weight}
            criterion = GatysEckerBethge2015PerceptualLoss(
                impl_params=impl_params, style_loss_kwargs=style_loss_kwargs
            )

            output_image = gatys_ecker_bethge_2015_nst(
                content_image,
                style_image,
                NUM_STEPS,
                impl_params=impl_params,
                criterion=criterion,
                quiet=quiet,
                logger=logger,
            )

            output_file = path.join(
                results_folder, "fig_3__{}__{}.jpg".format(row_label, column_label)
            )
            logger.sep_message(f"Saving result to {output_file}", bottom=False)
            write_image(output_image, output_file)


if __name__ == "__main__":
    device = None
    quiet = False

    images_root = utils.get_images_root()
    source_folder = path.join(images_root, "source")
    results_root = path.join(
        images_root, "results", path.splitext(path.basename(__file__))[0]
    )
    device = utils.parse_device(device)
    logger = utils.get_default_logger()

    with utils.log_replication_info(
        logger,
        title="A Neural Algorithm of Artistic Style",
        url="https://arxiv.org/abs/1508.06576",
        author="Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge",
        year=2015,
    ):

        for impl_params in (True, False):
            results_folder = path.join(
                results_root, "implementation" if impl_params else "paper"
            )
            figure_2(source_folder, results_folder, device, impl_params, logger, quiet)
            figure_3(source_folder, results_folder, device, impl_params, logger, quiet)
