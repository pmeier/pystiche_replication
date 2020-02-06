from os import path
import contextlib
import torch
from pystiche.image.transforms.functional import (
    resize,
    rgb_to_yuv,
    yuv_to_rgb,
    rgb_to_grayscale,
    grayscale_to_fakegrayscale,
    transform_channels_affinely,
)
from pystiche.cuda import abort_if_cuda_memory_exausts
from pystiche_replication.utils import read_image, read_guides, write_image
from pystiche_replication import gatys_et_al_2017_nst, gatys_et_al_2017_guided_nst
import utils


@contextlib.contextmanager
def replicate_figure(logger, figure, impl_params):
    params = "implementation" if impl_params else "paper"
    header = f"Replicating {figure} with {params} parameters"
    with logger.environ(header):
        yield


def log_saving_info(logger, output_file):
    logger.sep_message(f"Saving result to {output_file}", bottom=False)


def figure_2(
    source_folder, guides_root, replication_folder, device, impl_params, logger, quiet
):
    @abort_if_cuda_memory_exausts
    def figure_2_d(content_image, style_image):
        with replicate_figure(logger, "2 (d)", impl_params):
            output_image = gatys_et_al_2017_nst(
                content_image,
                style_image,
                impl_params=impl_params,
                quiet=quiet,
                logger=logger,
            )

            output_file = path.join(replication_folder, "fig_2__d.jpg")
            log_saving_info(logger, output_file)
            write_image(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_2_ef(
        label,
        content_image,
        content_house_guide,
        content_sky_guide,
        style_house_image,
        style_house_guide,
        style_sky_image,
        style_sky_guide,
    ):
        content_guides = {"house": content_house_guide, "sky": content_sky_guide}
        style_images_and_guides = {
            "house": (style_house_image, style_house_guide),
            "sky": (style_sky_image, style_sky_guide),
        }
        with replicate_figure(logger, f"2 ({label})", impl_params):

            output_image = gatys_et_al_2017_guided_nst(
                content_image,
                content_guides,
                style_images_and_guides,
                impl_params=impl_params,
                quiet=quiet,
                logger=logger,
            )

            output_file = path.join(replication_folder, f"fig_2__{label}.jpg")
            log_saving_info(logger, output_file)
            write_image(output_image, output_file)

    content_file = path.join(source_folder, "house_concept_tillamook.jpg")
    content_image = read_image(content_file, device=device)
    content_guides = read_guides(guides_root, content_file, device)

    style1_file = path.join(source_folder, "watertown.jpg")
    style1_image = read_image(style1_file, device=device)
    style1_guides = read_guides(guides_root, style1_file, device)

    style2_file = path.join(source_folder, "van_gogh__wheat_field_with_cypresses.jpg")
    style2_image = read_image(style2_file, device=device)
    style2_guides = read_guides(guides_root, style2_file, device)

    figure_2_d(content_image, style1_image)

    figure_2_ef(
        "e",
        content_image,
        content_guides["house"],
        content_guides["sky"],
        style1_image,
        style1_guides["house"],
        style1_image,
        style1_guides["sky"],
    )

    figure_2_ef(
        "f",
        content_image,
        content_guides["house"],
        content_guides["sky"],
        style1_image,
        style1_guides["house"],
        style2_image,
        style2_guides["sky"],
    )


def figure_3(source_folder, replication_folder, device, impl_params, logger, quiet):
    def calculate_channelwise_mean_covariance(image):
        batch_size, num_channels, height, width = image.size()
        num_pixels = height * width
        image = image.view(batch_size, num_channels, num_pixels)

        mean = torch.mean(image, dim=2, keepdim=True)

        image_centered = image - mean
        cov = torch.bmm(image_centered, image_centered.transpose(1, 2)) / num_pixels

        return mean, cov

    def msqrt(x):
        e, v = torch.symeig(x, eigenvectors=True)
        return torch.chain_matmul(v, torch.diag(e), v.t())

    def match_channelwise_statistics(input, target, method):
        input_mean, input_cov = calculate_channelwise_mean_covariance(input)
        target_mean, target_cov = calculate_channelwise_mean_covariance(target)

        input_cov, target_cov = [cov.squeeze(0) for cov in (input_cov, target_cov)]
        if method == "image_analogies":
            matrix = torch.mm(msqrt(target_cov), torch.inverse(msqrt(input_cov)))
        elif method == "cholesky":
            matrix = torch.mm(
                torch.cholesky(target_cov), torch.inverse(torch.cholesky(input_cov))
            )
        else:
            # FIXME: add error message
            raise ValueError
        matrix = matrix.unsqueeze(0)

        bias = target_mean - torch.bmm(matrix, input_mean)

        return transform_channels_affinely(input, matrix, bias)

    @abort_if_cuda_memory_exausts
    def figure_3_c(content_image, style_image):
        with replicate_figure(logger, "3 (c)", impl_params):
            output_image = gatys_et_al_2017_nst(
                content_image, style_image, impl_params=impl_params, quiet=True
            )

            output_file = path.join(replication_folder, "fig_3__c.jpg")
            log_saving_info(logger, output_file)
            write_image(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_3_d(content_image, style_image):
        content_image_yuv = rgb_to_yuv(content_image)
        content_luminance = grayscale_to_fakegrayscale(content_image_yuv[:, 0])
        content_chromaticity = content_image_yuv[:, 1:]

        style_luminance = grayscale_to_fakegrayscale(rgb_to_grayscale(style_image))

        with replicate_figure(logger, "3 (d)", impl_params):
            output_luminance = gatys_et_al_2017_nst(
                content_luminance, style_luminance, impl_params=impl_params, quiet=True
            )
            output_luminance = torch.mean(output_luminance, dim=1, keepdim=True)
            output_chromaticity = resize(
                content_chromaticity, output_luminance.size()[2:]
            )
            output_image_yuv = torch.cat((output_luminance, output_chromaticity), dim=1)
            output_image = yuv_to_rgb(output_image_yuv)

            output_file = path.join(replication_folder, "fig_3__d.jpg")
            log_saving_info(logger, output_file)
            write_image(output_image, output_file)

    @abort_if_cuda_memory_exausts
    def figure_3_e(content_image, style_image, method="cholesky"):
        style_image = match_channelwise_statistics(style_image, content_image, method)

        with replicate_figure(logger, "3 (e)", impl_params):
            output_image = gatys_et_al_2017_nst(
                content_image, style_image, impl_params=impl_params, quiet=True
            )

            output_file = path.join(replication_folder, "fig_3__e.jpg")
            log_saving_info(logger, output_file)
            write_image(output_image, output_file)

    content_file = path.join(
        source_folder, "janssen__schultenhof_mettingen_bauerngarten_8.jpg"
    )
    content_image = read_image(content_file).to(device)

    style_file = path.join(source_folder, "van_gogh__starry_night_over_rhone.jpg")
    style_image = read_image(style_file).to(device)

    figure_3_c(content_image, style_image)
    figure_3_d(content_image, style_image)
    figure_3_e(content_image, style_image)


if __name__ == "__main__":
    device = None
    quiet = False

    images_root = utils.get_images_root()
    source_folder = path.join(images_root, "source")
    guides_root = path.join(images_root, "guides")
    replication_root = path.join(
        images_root, "results", path.splitext(path.basename(__file__))[0]
    )
    device = utils.parse_device(device)
    logger = utils.get_default_logger()

    with utils.log_replication_info(
        logger,
        title="Controlling Perceptual Factors in Neural Style Transfer",
        url="http://openaccess.thecvf.com/content_cvpr_2017/papers/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.pdf",
        author="Leon Gatys et. al.",
        year=2017,
    ):
        for impl_params in (True, False):
            replication_folder = path.join(
                replication_root, "implementation" if impl_params else "paper"
            )

            figure_2(
                source_folder,
                guides_root,
                replication_folder,
                device,
                impl_params,
                logger,
                quiet,
            )
            figure_3(
                source_folder, replication_folder, device, impl_params, logger, quiet
            )
