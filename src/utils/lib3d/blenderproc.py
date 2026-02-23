import blenderproc as bproc
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os, sys
import ipdb
from PIL import Image
import trimesh


def render_blender_proc(
    cad_path,
    output_dir,
    obj_poses,
    img_size,
    intrinsic,
    tless_like,
):

    bproc.init()
    cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(
        np.eye(4), ["X", "-Y", "-Z"]
    )
    bproc.camera.add_camera_pose(cam2world)
    bproc.camera.set_intrinsics_from_K_matrix(intrinsic, img_size[1], img_size[0])
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([1, -1, 1])
    light.set_energy(200)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-1, -1, -1])
    light.set_energy(200)
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-1, 0, -1])
    light.set_energy(200)
    light.set_type("POINT")
    light.set_location([1, 0, 1])
    light.set_energy(200)

    # load the objects into the scene
    cad_id, model_id = cad_path.split("/")[-4:-2]
    shapeNet_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(cad_path)))
    )
    obj = bproc.loader.load_shapenet(
        shapeNet_path,
        used_synset_id=cad_id,
        used_source_id=model_id,
        move_object_origin=False,
    )
    # re-center object so that the center of 3D bounding box (0, 0, 0)
    bounding_box = obj.get_bound_box()
    center_box = np.mean(bounding_box, axis=0)
    obj.set_location(-center_box)

    if tless_like:
        # Check if the object has materials assigned
        obj.clear_materials()
        obj.new_material("tless_like")
        mat = obj.get_materials()[0]
        grey_col = np.random.uniform(0.2, 0.4)
        mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])

    # obj.set_origin(mode="CENTER_OF_VOLUME")  # CENTER_OF_MASS
    import bpy

    # print(dir(obj))
    obj.select()
    # bpy.data.objects['model_normalized'].select = True
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    # # bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    # bpy.ops.object.select_all(action='DESELECT')

    obj.set_cp("category_id", 1)
    # activate normal and distance rendering
    bproc.renderer.enable_depth_output(True)
    # set the amount of samples, which should be used for the color rendering
    bproc.renderer.set_max_amount_of_samples(10)
    bproc.renderer.set_output_format(enable_transparency=True)
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])

    if "query" in obj_poses:
        for name_pose in ["query", "ref"]:
            for idx_frame, pose in enumerate(obj_poses[name_pose]):
                obj.set_local2world_mat(pose)
                data = bproc.renderer.render()
                rgb = Image.fromarray(np.uint8(data["colors"][0])).convert("RGBA")
                rgb.save(os.path.join(output_dir, f"{idx_frame:06d}_{name_pose}.png"))
                depth = data["depth"][0]
                mask = data["category_id_segmaps"][0] == obj.get_cp(
                    "category_id"
                )  # 获取物体掩码
                depth[~mask] = 0  # 将非物体区域深度设为0
                depth[depth > 10] = 0
                depth = Image.fromarray(np.uint16(depth * 1000), mode="I;16")
                depth.save(
                    os.path.join(output_dir, f"{idx_frame:06d}_{name_pose}_depth.png")
                )
    else:
        template_poses = obj_poses["template_poses"]
        for idx_frame, pose in enumerate(template_poses):
            obj.set_local2world_mat(pose)
            data = bproc.renderer.render()
            rgb = Image.fromarray(np.uint8(data["colors"][0])).convert("RGBA")
            rgb.save(os.path.join(output_dir, f"{idx_frame:06d}.png"))
            depth = data["depth"][0]
            mask = data["category_id_segmaps"][0] == obj.get_cp(
                "category_id"
            )  # 获取物体掩码
            depth[~mask] = 0  # 将非物体区域深度设为0
            depth[depth > 10] = 0
            depth = Image.fromarray(np.uint16(depth * 1000), mode="I;16")
            depth.save(os.path.join(output_dir, f"{idx_frame:06d}_depth.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cad_path", nargs="?", help="Path to the model file")
    parser.add_argument("obj_dir", nargs="?", help="Path to pose")
    parser.add_argument("gpu_id", nargs="?", help="GPUs id")
    parser.add_argument("texture", nargs="?", help="tless_like or not")
    parser.add_argument("disable_output", nargs="?", help="Disable output of blender")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpu_id)

    tless_like = True if args.texture == "tless_like" else False
    poses = np.load(os.path.join(args.obj_dir, "poses.npz"))
    intrinsic = np.array([[262.5, 0.0, 128], [0.0, 262.5, 128], [0.0, 0.0, 1.0]])

    img_size = [256, 256]
    if args.disable_output == "true":
        # redirect output to log file
        logfile = os.path.join(args.obj_dir, "render.log")
        open(logfile, "a").close()
        old = os.dup(1)
        sys.stdout.flush()
        os.close(1)
        os.open(logfile, os.O_WRONLY)
    # scale_meter do not change the binary mask but recenter_origin change it
    render_blender_proc(
        args.cad_path,
        args.obj_dir,
        poses,
        intrinsic=intrinsic,
        img_size=img_size,
        tless_like=tless_like,
    )
    if args.disable_output == "true":
        # disable output redirection
        os.close(1)
        os.dup(old)
        os.close(old)
        os.system("rm {}".format(logfile))
