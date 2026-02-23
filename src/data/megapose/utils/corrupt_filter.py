import glob
from tqdm import tqdm
from joblib import Memory


MEMORY = Memory(f".cache/{__name__}")


@MEMORY.cache
def load_corrupted(list_dir):
    corrupt_keys = set([])
    files = glob.glob(f"{list_dir}/*.txt")
    for file in tqdm(files):
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                scene_id_im_id, gt_ids = line.strip().split(":")
                gt_ids = eval(gt_ids)
                scene_id, im_id = scene_id_im_id.split("_")
                scene_id = int(scene_id)
                im_id = int(im_id)
                for gt_id in gt_ids:
                    corrupt_keys.add((scene_id, im_id, gt_id))
    return corrupt_keys


class CorruptFilter:
    def __init__(self, list_dir):
        self.list_dir = list_dir
        self.corrupt_keys = load_corrupted(list_dir)

    def is_corrupt(self, scene_id, im_id, gt_id):
        return (int(scene_id), int(im_id), int(gt_id)) in self.corrupt_keys


# python -m src.data.datasets.megapose.utils.corrupt_filter
if __name__ == "__main__":
    list_dir = "/data2/datasets/MegaPose-GSO/GSO_broken_depth_maps"
    corrupt_filter = CorruptFilter(list_dir)
    print(corrupt_filter.is_corrupt(22722, 18, 6))
    print(corrupt_filter.is_corrupt(22722, 18, 9))
    print(corrupt_filter.is_corrupt(22722, 18, 14))
