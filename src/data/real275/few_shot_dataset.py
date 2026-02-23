import ipdb
import torch
from tqdm import tqdm
from src.data.bop.few_shot_dataset import BOPFewShot
from src.data.bop.instance_dataset import BOPInstanceDataset
from src.data.bop.scene_dataset import BOPSceneDataset
from src.data.collate_importer import default_collate_fn
from src.data.foundation_pose.few_shot_dataset import FoundationPoseFewShot
from src.data.megapose.obj_ds.bop_object_dataset import BOPObjectDataset
from src.data.megapose.obj_ds.real275_dataset import Real275ObjectDataset
from src.data.real275.instance_dataset import Real275InstanceDataset
from src.data.real275.scene_dataset import Real275SceneDataset
from src.models.coordar.visualization import show_batch

# python -m src.data.real275.few_shot_dataset
if __name__ == "__main__":
    dzi_config = dict(
        dzi_type="uniform", dzi_pad_scale=1.5, dzi_scale_ratio=0.0, dzi_shift_ratio=0.0
    )
    """
        custom dataset for real275
    """
    # scene_dataset = Real275SceneDataset("data/Real275")
    # print(len(scene_dataset))
    # obj_ds = Real275ObjectDataset("data/Real275")
    # instance_dataset = Real275InstanceDataset(
    #     scene_dataset, obj_ds, dzi_config=dzi_config
    # )

    """
        bop dataset for real275
    """
    scene_dataset = BOPSceneDataset(
        "data/Real275/BOP", "real275", split="test", split_type="real"
    )
    obj_ds = BOPObjectDataset("data/Real275/BOP/real275/models")
    instance_dataset = BOPInstanceDataset(scene_dataset, obj_ds, dzi_config=dzi_config)

    print(len(instance_dataset))

    few_shot_ds = FoundationPoseFewShot(
        instance_dataset,
        num_ref=1,
        num_samples=2000,
        split="all",
        no_cache=True,
        shuffle_indices=False,
        unique_queries=True,
    )

    test_targets = few_shot_ds.get_test_targets()

    print(len(test_targets))

    for item in tqdm(few_shot_ds):
        break

    data_loader = torch.utils.data.DataLoader(
        few_shot_ds,
        num_workers=8,
        collate_fn=default_collate_fn,
        batch_size=8,
    )

    for batch in tqdm(data_loader):
        pass
        # ipdb.set_trace()
        show_batch(batch)
