import json
import math
import os
import ipdb
import numpy as np
from PIL import Image
import argparse
import sys
import asyncio

from tqdm import tqdm, trange
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import pxr
import omni.usd
from pxr import Gf
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import World
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
import isaacsim.core.utils.prims as prims_utils
import isaacsim.core.utils.bounds as bounds_utils
import isaacsim.core.utils.xforms as xform_utils
import isaacsim.core.utils.camera_utils as camera_utils
import isaacsim.core.utils.rotations as rotation_utils
import isaacsim.core.utils.viewports as viewport_utils
from isaacsim.sensors.camera import CameraView
import omni.replicator.core as rep
from pxr import UsdLux, Sdf
from omni.isaac.core.objects.ground_plane import GroundPlane


async def convert(in_file, out_file, load_materials=False):
    # This import causes conflicts when global
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    converter_context.ignore_materials = not load_materials
    # converter_context.ignore_animation = False
    # converter_context.ignore_cameras = True
    # converter_context.single_mesh = True
    # converter_context.smooth_normals = True
    # converter_context.preview_surface = False
    # converter_context.support_point_instancer = False
    # converter_context.embed_mdl_in_usd = False
    # converter_context.use_meter_as_world_unit = True
    # converter_context.create_world_as_default_root_prim = False
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(
        in_file, out_file, progress_callback, converter_context
    )
    success = True
    while True:
        success = await task.wait_until_finished()
        if not success:
            await asyncio.sleep(0.1)
        else:
            break
    return success


def set_prim_pose(prim, pose):
    prim_path = prims_utils.get_prim_path(prim)
    """Set the pose of a prim in the stage"""
    # xform_utils.reset_and_set_xform_ops(
    #     prim,
    #     translation=pxr.Gf.Vec3d(pose[:3, 3].tolist()),
    #     orientation=pxr.Gf.Quatd(
    #         *(rotation_utils.rot_matrix_to_quat(pose[:3, :3])).tolist()
    #     ),
    #     scale=pxr.Gf.Vec3d(1.0, 1.0, 1.0),
    # ) not working
    prims_utils.set_prim_property(
        prim_path,
        property_name="xformOp:translate",
        property_value=Gf.Vec3d(pose[:3, 3].tolist()),
    )
    prims_utils.set_prim_property(
        prim_path,
        property_name="xformOp:orient",
        property_value=pxr.Gf.Quatf(
            *(rotation_utils.rot_matrix_to_quat(pose[:3, :3])).tolist()
        ),
    )


class IsaacSimRenderer:
    def __init__(self, img_size, intrinsics):

        self.img_size = img_size
        self.intrinsics = intrinsics

        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)

        # cube_1 = self.world.scene.add(
        #     DynamicCuboid(
        #         prim_path="/new_cube_1",
        #         name="cube_1",
        #         position=np.array([0, 0, 1.0]),
        #         scale=np.array([1.0, 1.0, 1.0]),
        #         size=1.0,
        #         color=np.array([255, 0, 0]),
        #     )
        # )
        # self.world.scene.add_default_ground_plane()
        # GroundPlane(prim_path="/World/GroundPlane", z_position=0)

        # Set up camera
        self.setup_camera()
        # Set up lighting
        self.setup_lighting()

    def setup_task(self, cad_path, output_dir, obj_poses, tless_like):
        self.cad_path = cad_path
        self.output_dir = output_dir
        self.obj_poses = obj_poses
        self.tless_like = tless_like

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load the object
        self.load_object()

    def clean_object(self):
        """Clean up the object in the scene"""
        # Remove the object from the stage
        stage = omni.usd.get_context().get_stage()
        stage.RemovePrim(self.obj.GetPath())
        self.obj = None

    def setup_camera(self):
        """Set up camera with specified intrinsics"""
        # Convert K matrix to Isaac Sim camera parameters
        ((fx, _, cx), (_, fy, cy), (_, _, _)) = self.intrinsics
        width, height = self.img_size
        aspect_ratio = width / height
        distortion_coefficients = [0, 0, 0, 0, 0, 0, 0, 0]

        # Isaac Sim 基于 USD，USD 中的标准变换顺序是：
        # SRT 顺序：先缩放（Scale），再旋转（Rotate），最后平移（Translate）。
        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array(
                [0.0, 0.0, 0.0]
            ),  # 1 meter away from the side of the cube
            frequency=10,
            resolution=(width, height),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 0, 0]), degrees=True  # ZYX order
            ),
        )

        self.world.reset()
        self.camera.initialize()

        focal_length = 36
        aperture = focal_length * width / self.intrinsics[0, 0]

        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        self.camera.set_focal_length(focal_length)
        self.camera.set_horizontal_aperture(aperture)
        self.camera.set_vertical_aperture(aperture)
        self.camera.set_clipping_range(0.0001, 1.0e5)

        print(self.camera.get_intrinsics_matrix())

        # Set camera position (looking at origin)
        # self.camera.set_world_pose(position=[0, 0, 0], orientation=[1, 0, 0, 0])
        self.camera.add_motion_vectors_to_frame()
        self.camera.add_distance_to_image_plane_to_frame()
        self.camera.add_instance_segmentation_to_frame()

        self.render_product = rep.create.render_product(
            self.camera.prim_path,
            self.img_size,
        )
        self.rgb_out = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_out.attach([self.render_product])

        self.depth_out = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.depth_out.attach([self.render_product])

    def setup_lighting(self):
        stage = omni.usd.get_context().get_stage()
        dome_light_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path(dome_light_path))
        dome_light.CreateIntensityAttr(1000)

    def load_object(self):
        """Load the object from CAD path"""
        # convert to usd
        asset_path = f"{self.output_dir}/converted.usd"
        status = asyncio.get_event_loop().run_until_complete(
            convert(self.cad_path, asset_path, True)
        )
        if not status:
            print(f"ERROR Status is {status}")

        self.obj = add_reference_to_stage(asset_path, "/World/MainObject")
        # Set object pose
        prim_path = prims_utils.get_prim_path(self.obj)
        # Center the object (similar to BlenderProc's bounding box centering)
        cache = bounds_utils.create_bbox_cache()
        self.obj_centroid, axes, half_extent = bounds_utils.compute_obb(
            cache, prim_path=prim_path
        )

    def render_poses(self):
        """Render images for all poses"""
        for idx_frame, pose in tqdm(
            enumerate(self.obj_poses["template_poses"]), file=sys.stdout
        ):
            self.render_frame(pose, idx_frame)

    def get_obs(self):
        res = {
            "rgb": self.rgb_out.get_data(),
            "distance_to_image_plane": self.depth_out.get_data(),
        }
        return res

    def render_frame(self, pose, idx_frame):
        """Render a single frame with given pose"""
        # to keep consistent with blenderproc
        rx90 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        to_center = np.eye(4)
        to_center[:3, 3] = -self.obj_centroid
        obj_pose = pose @ rx90 @ to_center
        # isaac sim 的相机坐标有两种模式，一种是usd模式，和图形学传统模式一致，一种是world模式，相机朝x轴正方向
        # 这里转成opencv的相机坐标系
        self.camera.set_world_pose(
            position=np.array([0, 0, 0]),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([180, 0, 0]), degrees=True
            ),
            camera_axes="usd",
        )
        set_prim_pose(self.obj, obj_pose)

        # Update physics (needed for proper rendering)
        self.world.reset()
        simulation_app.update()
        while is_stage_loading():
            simulation_app.update()
        frame_dict = self.get_obs()
        # Get camera data
        rgb = frame_dict["rgb"]
        depth = frame_dict["distance_to_image_plane"]

        # Process and save images
        prefix = f"{idx_frame:06d}"

        # Save RGB
        rgb_img = Image.fromarray((rgb).astype(np.uint8), "RGBA")
        rgb_img.save(os.path.join(self.output_dir, f"{prefix}_isaac_sim.png"))

        # Save depth (convert to millimeters)
        # remove inf
        depth[np.isinf(depth)] = 0
        depth_mm = (depth * 1000).astype(np.uint16)
        depth_img = Image.fromarray(depth_mm, mode="I;16")
        depth_img.save(os.path.join(self.output_dir, f"{prefix}_depth_isaac_sim.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="?", help="Path to the task file")
    args = parser.parse_args()
    tasks = json.load(open(args.tasks))

    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)

    intrinsics = np.array([[525, 0.0, 256], [0.0, 525, 256], [0.0, 0.0, 1.0]])
    img_size = [512, 512]
    # Initialize and run renderer
    renderer = IsaacSimRenderer(
        img_size=img_size,
        intrinsics=intrinsics,
    )
    for task in tasks:
        tless_like = True if task["texture"] == "tless_like" else False
        poses = np.load(os.path.join(task["obj_dir"], "poses.npz"))
        renderer.setup_task(
            cad_path=task["cad_path"],
            output_dir=task["obj_dir"],
            obj_poses=poses,
            tless_like=tless_like,
        )
        renderer.render_poses()
        renderer.clean_object()

    # Wait for the data to be written to disk
    rep.orchestrator.wait_until_complete()


if __name__ == "__main__":
    main()
