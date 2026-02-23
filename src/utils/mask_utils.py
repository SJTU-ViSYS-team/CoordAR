import pycocotools.mask as cocomask
import numpy as np

def segmToRLE(segm, h, w):
    """Convert segmentation which can be polygons, uncompressed RLE to RLE.

    :return: binary mask (numpy 2D array)
    """
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = cocomask.frPyObjects(segm, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = cocomask.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def cocosegm2mask(segm, h, w):
    if isinstance(segm, np.ndarray):
        return segm
    rle = segmToRLE(segm, h, w)
    mask = rle2mask(rle, h, w)
    return mask


def binary_mask_to_rle(mask, compressed=True):
    assert mask.ndim == 2, mask.shape
    mask = mask.astype(np.uint8)
    if compressed:
        rle = cocomask.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
    else:
        rle = {"counts": [], "size": list(mask.shape)}
        counts = rle.get("counts")
        for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):  # noqa: E501
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
    return rle


def rle2mask(rle, height, width):
    if "counts" in rle and isinstance(rle["counts"], list):
        # if compact RLE, ignore this conversion
        # Magic RLE format handling painfully discovered by looking at the
        # COCO API showAnns function.
        rle = cocomask.frPyObjects(rle, height, width)
    mask = cocomask.decode(rle)
    return mask
