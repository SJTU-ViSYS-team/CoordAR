import argparse
import os
import shutil
import subprocess
from cv2 import imwrite
import ipdb
import numpy as np
from os.path import join
from mmcv.image.io import imread


from bop_toolkit_lib import inout
from bop_toolkit_lib.inout import save_ply
from bop_toolkit_lib.misc import calc_pts_diameter2
from src.data.bop.scene_dataset import BOPSceneDataset
from src.data.megapose.obj_ds.real275_dataset import Real275ObjectDataset
from src.data.real275.scene_dataset import Real275SceneDataset
from src.utils.misc import prepare_dir


def info_from_vertices(vertices: np.ndarray) -> dict:
    """Calculate object information from vertices."""
    mins, maxs = np.min(vertices, axis=0).tolist(), np.max(vertices, axis=0).tolist()
    vertices_sample = vertices[
        np.random.choice(
            vertices.shape[0], size=min(1000, vertices.shape[0]), replace=False
        )
    ]
    diameter = calc_pts_diameter2(vertices_sample).item()

    return {
        "diameter": diameter,
        "min_x": mins[0],
        "min_y": mins[1],
        "min_z": mins[2],
        "size_x": maxs[0] - mins[0],
        "size_y": maxs[1] - mins[1],
        "size_z": maxs[2] - mins[2],
    }


def save_models(obj_ds, bop_root):
    SYM = [{"axis": [0, 1, 0], "offset": [0, 0, 0]}]

    bop_models_root = os.path.join(bop_root, "models")
    if not os.path.exists(bop_models_root):
        os.makedirs(bop_models_root)

    models_info = dict()

    for obj_id in obj_ds.obj_ids:
        label = obj_ds.id2label[obj_id]
        obj = obj_ds.get_object_by_label(f"real275_{label}")
        model_p3d = obj.model_p3d
        vertices = model_p3d._verts_list[0].cpu().numpy() * 1000  # Convert to mm
        faces = model_p3d._faces_list[0].cpu().numpy()
        model = dict(
            pts=vertices,
            faces=faces,
        )
        save_ply(f"{bop_models_root}/obj_{obj_id:06d}.ply", model)

        # model info
        cur_info = info_from_vertices(vertices)
        if "bottle" in label or "bowl" in label or "can" in label:
            cur_info["symmetries_continuous"] = SYM
        models_info[str(obj_id)] = cur_info

    inout.save_json(join(bop_models_root, "models_info.json"), models_info)

    shutil.copytree(
        bop_models_root,
        join(bop_root, "models_eval"),
        dirs_exist_ok=True,
    )


def save_camera(bop_root):

    camera_path = join(bop_root, "camera.json")
    if not os.path.exists(camera_path):
        camera = {
            "depth_scale": 1.0,
            "width": 640,
            "height": 480,
            "fx": 591.0125,
            "fy": 590.16775,
            "cx": 322.525,
            "cy": 244.11084,
        }
        inout.save_json(camera_path, camera)


def save_data(scene_ds, obj_ds, split_root):
    gts = scene_ds.gts
    metas = scene_ds.metas
    label2id = obj_ds.label2id

    instance = 0
    for scene_id in gts.keys():
        scene_gt = gts[scene_id]
        for im_id in scene_gt.keys():
            # depth
            depth_src_path = f"{scene_ds.root_dir}/{scene_ds.split_type}_{scene_ds.split}/scene_{scene_id}/{im_id:04d}_depth.png"
            depth_tgt_path = f"{split_root}/{scene_id:06d}/depth/{im_id:06d}.png"
            prepare_dir(os.path.dirname(depth_tgt_path))
            shutil.copy(depth_src_path, depth_tgt_path)

            # rgb
            rgb_src_path = f"{scene_ds.root_dir}/{scene_ds.split_type}_{scene_ds.split}/scene_{scene_id}/{im_id:04d}_color.png"
            rgb_tgt_path = f"{split_root}/{scene_id:06d}/rgb/{im_id:06d}.png"
            prepare_dir(os.path.dirname(rgb_tgt_path))
            shutil.copy(rgb_src_path, rgb_tgt_path)

            # mask
            poses = scene_gt[im_id]
            mask_src_path = f"{scene_ds.root_dir}/{scene_ds.split_type}_{scene_ds.split}/scene_{scene_id}/{im_id:04d}_mask.png"
            mask_scene = imread(mask_src_path, "unchanged")
            for gt_id, pose in enumerate(poses):
                mask_single = mask_scene == (gt_id + 1)
                mask_tgt_path = f"{split_root}/{scene_id:06d}/mask_visib/{im_id:06d}_{gt_id:06d}.png"
                prepare_dir(os.path.dirname(mask_tgt_path))
                imwrite(mask_tgt_path, mask_single.astype(np.uint8) * 255)
            instance += 1

    print(f"Saved {instance} depth in {split_root}.")


def save_scene_camera(scene_ds, split_root):
    """Save camera parameters for each scene."""
    gts = scene_ds.gts

    K = np.array(
        [
            [591.0125, 0, 322.525],
            [0, 590.16775, 244.11084],
            [0, 0, 1],
        ]
    )
    for scene_id in gts.keys():
        cameras = {}
        for im_id in gts[scene_id].keys():
            cameras[im_id] = {
                "cam_K": K.reshape(-1).tolist(),
                "depth_scale": 1.0,
            }

        cameras_path = f"{split_root}/{scene_id:06d}/scene_camera.json"
        prepare_dir(os.path.dirname(cameras_path))
        inout.save_json(cameras_path, cameras)


def save_scene_gt(scene_ds, obj_ds, split_root):
    gts = scene_ds.gts
    metas = scene_ds.metas
    label2id = obj_ds.label2id

    instance = 0
    for scene_id in sorted(gts.keys()):
        scene_gt = gts[scene_id]
        data = {}
        for im_id in sorted(scene_gt.keys()):
            poses = scene_gt[im_id]
            for gt_id, pose in enumerate(poses):
                meta = metas[scene_id][im_id]
                obj_label = str(meta["name"][gt_id])
                obj_id = label2id[obj_label]
                R = pose[:3, :3]
                mesh_scale = np.sqrt((R.T @ R)[0][0])
                R = R / mesh_scale
                t = pose[:3, 3] * 1000
                item = dict(
                    cam_R_m2c=R.reshape(-1).tolist(),
                    cam_t_m2c=t.reshape(-1).tolist(),
                    obj_id=obj_id,
                )
                data.setdefault(im_id, []).append(item)
                instance += 1

        scene_gt_path = f"{split_root}/{scene_id:06d}/scene_gt.json"

        prepare_dir(os.path.dirname(scene_gt_path))
        inout.save_json(scene_gt_path, data)

    print(f"Saved {instance} instances in {split_root}.")


def save_test_targets(scene_ds, obj_ds, dataset_root, num_test=2000):
    """Save test targets for the real275 dataset."""
    gts = scene_ds.gts
    metas = scene_ds.metas
    label2id = obj_ds.label2id
    instances = []
    for scene_id in sorted(gts.keys()):
        scene_gt = gts[scene_id]
        for im_id in sorted(scene_gt.keys()):
            poses = scene_gt[im_id]
            for gt_id, pose in enumerate(poses):
                meta = metas[scene_id][im_id]
                obj_label = str(meta["name"][gt_id])
                obj_id = label2id[obj_label]
                instances.append(
                    dict(
                        scene_id=scene_id,
                        im_id=im_id,
                        obj_id=obj_id,
                        gt_id=gt_id,
                    )
                )

    instances = np.random.choice(instances, num_test, replace=False)

    # write test_bop_19.json
    instance_cnt = {}
    for instance in instances:
        scene_id = instance["scene_id"]
        im_id = instance["im_id"]
        obj_id = instance["obj_id"]
        gt_id = instance["gt_id"]

        instance_cnt.setdefault(scene_id, {}).setdefault(im_id, {}).setdefault(
            obj_id, []
        ).append(gt_id)

    test_target = []
    for scene_id, im_dict in sorted(instance_cnt.items()):
        for im_id, obj_dict in sorted(im_dict.items()):
            for obj_id, gt_ids in sorted(obj_dict.items()):
                test_target.append(
                    dict(
                        scene_id=scene_id,
                        im_id=im_id,
                        obj_id=obj_id,
                        inst_count=len(gt_ids),
                    )
                )
    inout.save_json(
        join(dataset_root, f"test_targets_bop19.json"),
        test_target,
    )
    print(
        f"Saved {len(test_target)} test targets in {dataset_root}/test_targets_bop19.json."
    )


# python -m src.scripts.real275.real2752bop
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate BOP-format data and annotations for real275 dataset."
    )
    parser.add_argument(
        "--root",
        type=str,
        help="Root directory of the real275 dataset.",
        default="./data/Real275",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to process (default: test).",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="real",
        help="Dataset split type to process (default: real).",
    )

    args = parser.parse_args()

    bop_root = os.path.join(args.root, "BOP")
    dataset_name = "real275"
    if not os.path.exists(f"{bop_root}/{dataset_name}"):
        os.makedirs(f"{bop_root}/{dataset_name}")

    obj_ds = Real275ObjectDataset(
        root_dir=args.root, split=args.split, split_type=args.split_type
    )

    scene_ds = Real275SceneDataset(
        root_dir=args.root, split=args.split, split_type=args.split_type
    )

    save_camera(f"{bop_root}/{dataset_name}")
    save_scene_gt(
        scene_ds, obj_ds, f"{bop_root}/{dataset_name}/{args.split}_{args.split_type}"
    )
    save_test_targets(scene_ds, obj_ds, f"{bop_root}/{dataset_name}", num_test=2000)
    save_scene_camera(
        scene_ds, f"{bop_root}/{dataset_name}/{args.split}_{args.split_type}"
    )

    save_data(
        scene_ds, obj_ds, f"{bop_root}/{dataset_name}/{args.split}_{args.split_type}"
    )
    save_models(obj_ds, f"{bop_root}/{dataset_name}")

    """ 
    calc gt masks, caution! this will overwrite the existing visib masks
    """

    # cmd = [
    #     "python",
    #     "src/third_party/bop_toolkit/scripts/calc_gt_masks.py",
    #     f"--datasets_path={bop_root}",
    #     f"--dataset={dataset_name}",
    #     f"--dataset_split={args.split}",
    #     f"--dataset_split_type={args.split_type}",
    # ]
    # result = subprocess.run(cmd, capture_output=False, text=True)
    # if result.returncode != 0:
    #     raise RuntimeError(f"Error running calc_gt_masks.py: {result.stderr}")

    """ 
    calc gt info
    """
    # cmd = [
    #     "python",
    #     "src/third_party/bop_toolkit/scripts/calc_gt_info.py",
    #     f"--datasets_path={bop_root}",
    #     f"--dataset={dataset_name}",
    #     f"--dataset_split={args.split}",
    #     f"--dataset_split_type={args.split_type}",
    # ]
    # result = subprocess.run(cmd, capture_output=False, text=True)
    # if result.returncode != 0:
    #     raise RuntimeError(f"Error running calc_gt_masks.py: {result.stderr}")

    # test
    BOPSceneDataset(bop_root, "real275", args.split, args.split_type)
