import importlib
from typing import Any, Dict
import ipdb
import torch
from pytorch3d.structures import Meshes, join_meshes_as_batch
import numpy as np
import traceback

from src.utils.misc import LD2DL
from src.utils.tensor_collection import to_device


class CollateImporter(object):
    def __init__(self, train: str, val: str, test: str, predict: str) -> None:
        self._train_fn = getattr(
            importlib.import_module(".".join(train.split(".")[:-1])),
            train.split(".")[-1],
        )
        self._val_fn = getattr(
            importlib.import_module(".".join(val.split(".")[:-1])),
            val.split(".")[-1],
        )

        self._test_fn = getattr(
            importlib.import_module(".".join(test.split(".")[:-1])),
            test.split(".")[-1],
        )
        self._predict_fn = getattr(
            importlib.import_module(".".join(predict.split(".")[:-1])),
            predict.split(".")[-1],
        )

    @property
    def train_fn(self) -> Any:
        return self._train_fn

    @property
    def val_fn(self) -> Any:
        return self._val_fn

    @property
    def test_fn(self) -> Any:
        return self._test_fn

    @property
    def predict_fn(self) -> Any:
        return self._predict_fn


class FunctionImporter(object):
    def __init__(self, signature: str) -> None:
        self.func = getattr(
            importlib.import_module(".".join(signature.split(".")[:-1])),
            signature.split(".")[-1],
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def default_collate_fn(data):
    tensors = LD2DL(data)
    for k, v in tensors.items():
        try:
            if k in ["bboxes", "occlusionRatio"]:
                tensor_list = v
                max_size = _max_by_axis([list(img.shape) for img in tensor_list])
                batch_shape = [len(tensor_list)] + max_size
                dtype = tensor_list[0].dtype
                tensor = np.zeros(batch_shape, dtype=np.float32)
                for t, pad_t in zip(tensor_list, tensor):
                    slices = tuple(slice(0, s) for s in t.shape)
                    pad_t[slices] = t
                tensors[k] = torch.from_numpy(tensor)
            elif k in ["symmetries"]:
                tensor_list = v
                max_size = _max_by_axis([list(img.shape) for img in tensor_list])
                batch_shape = [len(tensor_list)] + max_size
                dtype = tensor_list[0].dtype
                device = tensor_list[0].device
                tensor = torch.zeros(batch_shape, dtype=dtype, device=device)

                if len(batch_shape) == 4:
                    # (b, n_sym, 4, 4)
                    n_sym = max_size[0]
                    for img, pad_img in zip(tensor_list, tensor):
                        # img: (?, 4, 4)
                        pad_img[: img.shape[0], :, :].copy_(img)
                        # padding with last element
                        for i in range(img.shape[0], n_sym):
                            pad_img[i, :, :].copy_(img[-1])
                else:
                    # or (b, m, n_sym, 4, 4)
                    n_sym = max_size[1]
                    for img, pad_img in zip(tensor_list, tensor):
                        # img: (m, ?, 4, 4)
                        pad_img[:, : img.shape[1], :, :].copy_(img)
                        for i in range(img.shape[1], n_sym):
                            pad_img[:, i, :, :].copy_(img[:, -1])
                tensors[k] = tensor
            elif isinstance(v[0], Meshes):
                tensors[k] = join_meshes_as_batch(v)
            elif isinstance(v[0], np.ndarray):
                v = [torch.from_numpy(item) for item in v]
                tensors[k] = torch.stack(v, 0)
            elif isinstance(v[0], torch.Tensor):
                tensors[k] = torch.stack(v, 0)
            elif isinstance(v[0], str):
                tensors[k] = v
            elif isinstance(v[0], int):
                tensors[k] = torch.tensor(v)
            elif isinstance(v[0], float):
                tensors[k] = torch.tensor(v, dtype=torch.float32)
            elif isinstance(
                v[0], (np.float32, np.float64, np.int32, np.int64)
            ):  # Handle numpy numeric types
                tensors[k] = torch.tensor(np.array(v))

        except Exception as e:
            print(f"{k}, {v}")
            traceback.print_exc()
            raise e
    return tensors


def preprocess_batch(batch: Dict[Any, Any], debug=False, stage="train"):
    batch = to_device(batch, "cuda")
    return batch


if __name__ == "__main__":
    importer = CollateImporter("utils.coco.misc.collate_fn")
    print(importer.get_collate_fn())
