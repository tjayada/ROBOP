import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from loaders.utils import _pil_loader

log = logging.getLogger(__name__)


def find_ndds_data_in_dir(
    input_dir, data_extension="json", image_extension=None, requested_image_types="all",
):
    input_dir = os.path.expanduser(input_dir)
    assert os.path.exists(
        input_dir
    ), 'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = os.listdir(input_dir)

    assert isinstance(
        data_extension, str
    ), 'Expected "data_extension" to be a string, but it is "{}".'.format(
        type(data_extension)
    )
    data_full_ext = "." + data_extension

    if image_extension is None:
        image_exts_to_try = ["png", "jpg"]
        num_image_exts = []
        for image_ext in image_exts_to_try:
            num_image_exts.append(len([f for f in dirlist if f.endswith(image_ext)]))
        max_num_image_exts = np.max(num_image_exts)
        idx_max = np.where(num_image_exts == max_num_image_exts)[0]
        image_extension = image_exts_to_try[idx_max[0]]
        if len(idx_max) > 1 and max_num_image_exts > 0:
            log.warning(
                'Multiple sets of images detected in NDDS dataset with different extensions. Using extension "%s".',
                image_extension,
            )
    else:
        assert isinstance(
            image_extension, str
        ), 'If specified, expected "image_extension" to be a string, but it is "{}".'.format(
            type(image_extension)
        )
    image_full_ext = "." + image_extension

    assert (
        requested_image_types is None
        or requested_image_types == "all"
        or isinstance(requested_image_types, list)
    ), "Expected \"requested_image_types\" to be None, 'all', or a list of requested_image_types."

    data_filenames = [f for f in dirlist if f.endswith(data_full_ext)]
    data_filenames.sort()

    data_names = [os.path.splitext(f)[0] for f in data_filenames if f[0].isdigit()]

    if not data_names:
        return None, None

    data_paths = [os.path.join(input_dir, f) for f in data_filenames if f[0].isdigit()]

    # RGB only: ROBOP requests ["rgb"] and panda-orb ships no depth/cs images.
    if requested_image_types == "all":
        first_entry_name = data_names[0]
        find_rgb = (first_entry_name + ".rgb" + image_full_ext) in dirlist

    elif requested_image_types:
        for this_image_type in requested_image_types:
            assert (
                this_image_type == "rgb"
            ), 'Image type "{}" not recognized.'.format(this_image_type)
        find_rgb = "rgb" in requested_image_types

    else:
        find_rgb = False

    dict_of_lists_images = {}
    n_samples = len(data_names)

    if find_rgb:
        rgb_paths = [
            os.path.join(input_dir, f + ".rgb" + image_full_ext) for f in data_names
        ]
        for n in range(n_samples):
            assert os.path.exists(
                rgb_paths[n]
            ), 'Expected image "{}" to exist, but it does not.'.format(rgb_paths[n])
        dict_of_lists_images["rgb"] = rgb_paths

    found_images = [
        dict(zip(dict_of_lists_images, t)) for t in zip(*dict_of_lists_images.values())
    ]

    dict_of_lists = {"name": data_names, "data_path": data_paths}
    if find_rgb:
        dict_of_lists["image_paths"] = found_images

    found_data = [dict(zip(dict_of_lists, t)) for t in zip(*dict_of_lists.values())]

    found_configs = {"camera": None, "object": None, "unsorted": []}
    data_filenames_without_images = [f for f in data_filenames if not f[0].isdigit()]

    for data_filename in data_filenames_without_images:
        if data_filename == "_camera_settings" + data_full_ext:
            found_configs["camera"] = os.path.join(input_dir, data_filename)
        elif data_filename == "_object_settings" + data_full_ext:
            found_configs["object"] = os.path.join(input_dir, data_filename)
        else:
            found_configs["unsorted"].append(os.path.join(input_dir, data_filename))

    return found_data, found_configs


def load_camera_parameters(data_folder: str) -> tuple:
    """
    Load camera intrinsics from NDDS/DREAM _camera_settings.json.

    Returns:
        (fx, fy, cx, cy) in pixels.
    """
    _, ndds_data_configs = find_ndds_data_in_dir(data_folder)
    if ndds_data_configs["camera"] is None:
        raise FileNotFoundError(
            f"No _camera_settings.json found in {data_folder}. "
            "Expected NDDS/DREAM layout with digit-prefixed .json and _camera_settings.json."
        )
    with open(ndds_data_configs["camera"], "r") as json_file:
        data = json.load(json_file)
    fx = data["camera_settings"][0]["intrinsic_settings"]["fx"]
    fy = data["camera_settings"][0]["intrinsic_settings"]["fy"]
    cx = data["camera_settings"][0]["intrinsic_settings"]["cx"]
    cy = data["camera_settings"][0]["intrinsic_settings"]["cy"]
    return fx, fy, cx, cy


class DREAMDataset(Dataset):
    """
    DREAM-format Panda (or similar NDDS) dataset: digit-prefixed .json + .rgb.jpg,
    _camera_settings.json. Compatible with evaluate_ADD via get_data_with_keypoints.
    """

    def __init__(self, data_folder: str, trans_to_tensor):
        self.data_folder = os.path.expanduser(data_folder)
        self.trans_to_tensor = trans_to_tensor

        found, _ = find_ndds_data_in_dir(self.data_folder, requested_image_types=["rgb"])
        if found is None:
            raise FileNotFoundError(f"No NDDS data in {self.data_folder}")
        self._ndds_list = found

    def __len__(self) -> int:
        return len(self._ndds_list)

    def _load_sample(self, idx: int):
        data_sample = self._ndds_list[idx]
        img_path = data_sample["image_paths"]["rgb"]
        image_pil = _pil_loader(img_path)
        image = self.trans_to_tensor(image_pil)
        with open(data_sample["data_path"], "r") as f:
            data = json.load(f)
        joints = data["sim_state"]["joints"]
        joint_angle = torch.tensor(
            [joints[i]["position"] for i in range(7)], dtype=torch.float32
        )
        return image, joint_angle, data

    def get_data_with_keypoints(self, idx: int):
        """Same interface as CtRNet ImageDataLoaderReal.get_data_with_keypoints."""
        image, joint_angle, data = self._load_sample(idx)
        keypoints = data["objects"][0]["keypoints"]
        return image, joint_angle, keypoints
