from os import path
from pystiche.cuda import abort_if_cuda_memory_exausts
from pystiche_replication import li_wand_2016_nst
from pystiche_replication.utils import read_image, write_image
import utils


@abort_if_cuda_memory_exausts
def figure_6(source_folder, replication_folder, device, impl_params):
    content_files = ("jeffrey_dennard.jpg", "theilr__s.jpg")
    style_files = ("picasso__self-portrait_1907.jpg", "kandinsky__composition_viii.jpg")
    locations = ("top", "bottom")

    for content_file, style_file, location in zip(
        content_files, style_files, locations
    ):
        content_image = read_image(
            path.join(source_folder, content_file), device=device
        )
        style_image = read_image(path.join(source_folder, style_file), device=device)

        params = "implementation" if impl_params else "paper"
        print(f"Replicating the {location} half of figure 6 with {params} parameters")
        output_image = li_wand_2016_nst(
            content_image, style_image, impl_params, quiet=False
        )

        output_file = path.join(replication_folder, "fig_6__{}.jpg".format(location))
        print(f"Saving result to {output_file}")
        write_image(output_image, output_file)


if __name__ == "__main__":
    device = None

    images_root = utils.get_images_root()
    source_folder = path.join(images_root, "source")
    replication_root = path.join(
        images_root, "results", path.splitext(path.basename(__file__))[0]
    )
    device = utils.parse_device(device)

    utils.print_replication_info(
        title="Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis",
        url="https://ieeexplore.ieee.org/document/7780641",
        author="Chuan Li and Michael Wand",
        year=2016,
    )
    for impl_params in (True, False):
        replication_folder = path.join(
            replication_root, "implementation" if impl_params else "paper"
        )

        figure_6(source_folder, replication_folder, device, impl_params)
        utils.print_sep_line()
