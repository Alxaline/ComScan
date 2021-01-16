# http://nipy.org/nipy/api/generated/nipy.labs.mask.html
#
# nipy.labs.mask.intersect_masks(input_masks, output_filename=None, threshold=0.5, cc=True)
#
# Given
# a
# list
# of
# input
# mask
# images, generate
# the
# output
# image
# which is the
# the
# threshold - level
# intersection
# of
# the
# inputs
import os
import re
from typing import List, Tuple, Union, Optional, Callable

import numpy as np
from tqdm import tqdm

from harmonization.utils import load_nifty_volume_as_array
from harmonization.utils import mat_to_bytes

list_file = [
    "/media/acarre/Data/data_stock/BraTS/BraTS2020_Training/BraTS20_Training_001/BraTS20_Training_001_t1.nii.gz",
    "/media/acarre/Data/data_stock/BraTS/BraTS2020_Training/BraTS20_Training_002/BraTS20_Training_002_t1.nii.gz"]
from nipy.labs.mask import compute_mask_files


def _compute_mask_files(input_path: List[str],
                        output_path: Optional[str] = None,
                        return_mean: bool = False,
                        m: float = 0.2,
                        M: float = 0.9,
                        cc: int = 1,
                        exclude_zeros: bool = False,
                        opening: int = 2) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Wrap the nipy compute mask files function.


    Compute a mask file from MRI nifti file(s)

    Compute and write the mask of an image based on the grey level
    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    m and M of the total image histogram.

    In case of failure, it is usually advisable to increase m.

    :param input_path: string. list of filenames (3D).
    :param output_path: string or None, optional
        path to save the output nifti image (if not None).
    :param return_mean: boolean, optional
        if True, and output_filename is None, return the mean image also, as
        a 3D array (2nd return argument).
    :param m: float, optional
        lower fraction of the histogram to be discarded.
    :param M: float, optional
        upper fraction of the histogram to be discarded.
    :param cc: boolean, optional
        if cc is True, only the largest connect component is kept.
    :param exclude_zeros: boolean, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.
    :param opening: int, optional
        Size of the morphological opening performed as post-processing
    :return:
     mask : 3D boolean array
        The brain mask
    mean_image : 3d ndarray, optional
        The main of all the images used to estimate the mask. Only
        provided if `return_mean` is True.
    """

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = compute_mask_files(input_filename=input_path, output_filename=output_path,
                                return_mean=return_mean, m=m, M=M, cc=cc, exclude_zeros=exclude_zeros, opening=opening)

    if return_mean:
        mask, mean_volume = output[0], output[1]
    else:
        mask, mean_volume = output, None

    # nibabel array is x,y,z and sitk is z,y,x -> convert nibabel to sitk
    mask = np.swapaxes(mask, 0, 2)

    return mask, mean_volume


def flatten_nifti_files(input_path: List[str], mask: Union[str, np.ndarray],
                        output_flattened_array_path: str = 'flattened_array',
                        dtype: [np.dtype, Callable] = np.float16, save: bool = True,
                        compress_save: bool = True):
    """
    Flattened list of nifti files to a flattened array [n_images, n_masked_voxels] and save to .npy or .npz if compressed
    :param input_path: List of nifti files path
    :param mask: path of mask or array
    :param output_flattened_array_path: path of the output flattened array. No extension is needed. Will be save as
        .npy if no compression, else .npz
    :param save: save the flattened array
    :param dtype: dtype of the output flattened array. Default is float 16 to save memory
    :param compress_save: If true compress the numpy array into .npz
    :return: flattened array [n_images, n_masked_voxels]
    """
    if isinstance(mask, str):
        mask, _ = load_nifty_volume_as_array(input_path_file=mask)

    if not isinstance(dtype(1), np.floating):
        raise ValueError("dtype need to be float type")

    logical_mask = mask == 1  # force the mask to be logical type
    n_voxels_flattened = np.sum(logical_mask)
    n_images = len(input_path)

    required_memory = mat_to_bytes(nrows=n_images, ncols=n_voxels_flattened,
                                   dtype=int(re.findall('\d+', np.dtype(dtype).name)[0]), out="MB")
    print(f"required memory in RAM is: {required_memory:.2f} MB")

    flattened_array = np.zeros((n_images, n_voxels_flattened)).astype(dtype)
    for i, image_path in enumerate(tqdm(input_path, desc="Flattened array")):
        image_arr, _ = load_nifty_volume_as_array(image_path)
        flattened_array[i, :] = image_arr[logical_mask]

    if save:
        if not os.path.exists(os.path.dirname(output_flattened_array_path)):
            os.makedirs(os.path.dirname(output_flattened_array_path), exist_ok=True)
        if compress_save:
            np.savez_compressed(output_flattened_array_path, flattened_array)
        else:
            np.save(output_flattened_array_path, flattened_array)

    return flattened_array
