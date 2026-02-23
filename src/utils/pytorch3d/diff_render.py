from typing import List, Tuple, Union
import warnings
from joblib import Memory
from src.utils.trimesh_utils import as_mesh
import torch
import torch.nn as nn
import copy
import numpy as np
import trimesh
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import _read_header, MeshGlbFormat

import open3d as o3d
import ipdb
from functools import lru_cache
import matplotlib.pyplot as plt

# Data structures and functions for rendering
import pytorch3d.io as pio
from pytorch3d.structures import Meshes
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardFlatShader,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,
    Textures,
)

from src.utils.pytorch3d.light_free_shader import LightFreeShader

det = lambda t: np.linalg.det(t.detach().cpu().numpy())

CACHE_DIR = "./.cache/diff_render"

MEMORY = Memory(CACHE_DIR, verbose=0)


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf, fragments.pix_to_face


def render_posed_object(
    meshes: Meshes,
    TCO: torch.Tensor,
    K: torch.Tensor,
    resolution: Union[int, List[int], Tuple[int, int]] = (240, 320),
    re_render_iters=1,
    scale_res=1.0,
    shader_type="soft_phong",
    with_alpha_channel=True,
    dim_order="torch",
    light_loc=(0, 0, 100),
    znear=1.0,
    zfar=1000,
    pixel2face=False,
):
    """
    TCO: (b, 4, 4)
    K: (b, 3, 3)
    resolution: (b, 2) H,W order
    """
    batch_size = len(TCO)
    device = TCO.device
    assert (
        isinstance(resolution, int)
        or isinstance(resolution, tuple)
        or isinstance(resolution, list)
    )
    if isinstance(resolution, tuple) or isinstance(resolution, list):
        resolutions = torch.tensor(list([resolution])).int().to(device)
    else:
        resolutions = resolution
    assert resolutions.dtype == torch.int

    if len(meshes) == 1:
        meshes = join_meshes_as_batch([meshes[0] for i in range(batch_size)])

    lights = setup_lights(TCO, device, loc=light_loc)
    # lights = setup_ambient_light(device)

    materials = setup_materials(device)
    cameras = setup_cameras(
        K, TCO, resolutions, batch_size, scale_res, device, znear, zfar
    )
    raster_settings = setup_raster_settings(resolution)

    renderer = setup_renderer(
        cameras, raster_settings, lights, materials, shader_type, device
    )

    for itr in range(re_render_iters):
        rgb, depth, pix_to_face = renderer(meshes, lights=lights)
        # rgb = renderer(meshes, cameras=cameras)

        if rgb.max() > 1.1:
            warnings.warn(
                f"The rendered image is too bright! ({itr}) {rgb.mean().item()} {rgb.max().item()}"
            )
            rgb.clamp_(min=0.0, max=1.0)

        if (
            not torch.isnan(rgb).any()
            and (depth.amax(dim=[1, 2, 3]) > 0).all()
            and rgb.max() < 1.1
        ):
            break
    # rgb.requires_grad = True
    rgb, pix_to_face = post_process(meshes, rgb, depth, pix_to_face)

    rgb, depth, pix_to_face = (
        rgb.to(device=device),
        depth.to(device=device),
        pix_to_face.to(device=device),
    )
    if not with_alpha_channel:
        rgb = rgb[..., :3]
    if dim_order == "torch":
        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)
    elif dim_order == "tensorflow":
        pass
    else:
        raise NotImplementedError(f"dim_order {dim_order} not supported")
    if pixel2face:
        return rgb, depth, pix_to_face
    return rgb, depth


def setup_lights(TCO, device, loc):
    R_orig, T = TCO[..., :3, :3].contiguous(), TCO[..., :3, 3].contiguous()
    assert np.abs(det(R_orig) - 1).max() < 1e-1, det(R_orig)
    R = R_orig.transpose(1, 2)

    orig = torch.tensor([0, 0, 0]).to(R).view(1, -1, 1)
    light_loc = torch.tensor(loc).to(R).view(1, -1, 1)
    light_loc = (R @ light_loc).view(-1, 3)
    lights = PointLights(
        device=device,
        location=light_loc,
        diffuse_color=((0.5,) * 3,),
        ambient_color=((0.5,) * 3,),
        specular_color=((0.0,) * 3,),
    )
    return lights


def setup_ambient_light(device):
    lights = AmbientLights(device=device)
    return lights


def setup_cameras(
    K, poses, resolution, batch_size, scale_res, device, znear=0.1, zfar=1000
):
    R_orig, T = poses[..., :3, :3].contiguous(), poses[..., :3, 3].contiguous()
    assert np.abs(det(R_orig) - 1).max() < 1e-1, det(R_orig)
    R = R_orig.transpose(1, 2)

    principal_point = K[:, :2, 2] * scale_res
    focal_length = torch.stack((K[:, 0, 0], K[:, 1, 1]), dim=-1) * scale_res
    resolution = resolution.mul(scale_res).long()

    K_kwargs = dict(
        focal_length=-focal_length,
        in_ndc=False,
        image_size=resolution,
        principal_point=principal_point,
    )
    assert np.abs(det(R) - 1).max() < 1e-1, det(R)
    cameras = PerspectiveCameras(device=device, R=R, T=T, **K_kwargs)

    # phong shader default zfar is 100
    setattr(cameras, "zfar", zfar)
    setattr(cameras, "znear", znear)

    assert len(cameras) == batch_size, [len(cameras), batch_size]
    return cameras


def setup_materials(device):
    ambient_color = torch.tensor(((1, 1, 1),))
    diffuse_color = torch.tensor(((1, 1, 1),))
    # disable specular
    specular_color = torch.tensor(((1, 1, 1),)) * 0
    shininess = 64
    materials = Materials(
        device=device,
        ambient_color=ambient_color,
        diffuse_color=diffuse_color,
        specular_color=specular_color,
        shininess=shininess,
    )
    return materials
    # return Materials(device=device)


def setup_raster_settings(image_size: Tuple[int, int]):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=False,
    )
    # Slightly slower but necessary
    raster_settings.max_faces_per_bin = int(1e5)

    return raster_settings


def denormalize(verts, center, extent):
    return verts * extent[None] + center[None]


def normalize_verts(vertices):
    norm_verts = copy.deepcopy(vertices)
    xmin, xmax = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    ymin, ymax = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    scale = max(max(xmax - xmin, ymax - ymin), zmax - zmin) / 2.0

    norm_verts[:, 0] -= (xmax + xmin) / 2.0
    norm_verts[:, 1] -= (ymax + ymin) / 2.0
    norm_verts[:, 2] -= (zmax + zmin) / 2.0
    norm_verts[:, :3] /= scale
    return norm_verts


def normalize_mesh_(mesh):
    mesh.vertices = normalize_verts(mesh.vertices)
    # ipdb.set_trace()
    # AABB = mesh.bounds
    # center = np.mean(AABB, axis=0)
    # scale = np.max(AABB[1] - AABB[0]) / 2.0
    # mesh.vertices -= center
    # mesh.vertices /= scale


# @MEMORY.cache
def setup_meshes(
    cad_path,
    scale=1.0,
    normalize=False,
    disable_textures=False,
):
    if cad_path[-4:] == ".obj":
        try:
            verts, faces, aux = pio.load_obj(cad_path)
        except Exception as e:
            verts, faces, aux = pio.load_obj(cad_path, load_textures=False)
            print(f"Error loading {cad_path}: {e}")
        if normalize:
            verts = normalize_verts(verts)
        tex_maps = aux.texture_images
        if tex_maps:
            faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
            verts_uvs = aux.verts_uvs[None, ...]
            texture_image = list(tex_maps.values())[0]
            texture_image = texture_image[None, ...]  # (1, H, W, 3)
        else:
            verts_uvs = torch.zeros(1, verts.shape[0], 2).to(verts)
            faces_uvs = torch.zeros(1, faces.verts_idx.shape[0], 3).to(faces.verts_idx)
            texture_image = torch.ones(1, 5, 5, 3).to(verts)
        if disable_textures:
            verts_rgb_colors = torch.ones(verts.shape).to(verts)
            textures = Textures(verts_rgb=verts_rgb_colors[None])
        else:
            textures = TexturesUV(
                verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image
            )
        meshes = Meshes(verts=[verts], faces=[faces.verts_idx], textures=textures)
    else:
        if cad_path[-4:] == ".glb":
            mesh = trimesh.load(cad_path, force="mesh")
            if normalize:
                normalize_mesh_(mesh)
            verts = torch.from_numpy(np.array(mesh.vertices)).float()
            faces_idx = torch.from_numpy(np.array(mesh.faces))
            verts_rgb_colors = torch.ones(verts.shape).to(verts)
            textures = Textures(verts_rgb=verts_rgb_colors[None])
            meshes = Meshes(verts=[verts], faces=[faces_idx], textures=textures)
        elif cad_path[-4:] == ".ply":
            mesh = o3d.io.read_triangle_mesh(cad_path)
            verts = torch.from_numpy(np.array(mesh.vertices)).float()
            if normalize:
                verts = normalize_verts(verts)
            faces_idx = torch.from_numpy(np.array(mesh.triangles))
            verts_rgb_colors = torch.from_numpy(np.array(mesh.vertex_colors)).float()
            if len(verts_rgb_colors) == 0:
                verts_rgb_colors = torch.ones(verts.shape).to(verts)
            textures = Textures(verts_rgb=verts_rgb_colors[None])
            meshes = Meshes(verts=[verts], faces=[faces_idx], textures=textures)
    meshes.scale_verts_(scale)
    return meshes


def setup_renderer(cameras, raster_settings, lights, materials, shader_type, device):
    shaders = {
        "flat": HardFlatShader,
        "soft_phong": SoftPhongShader,
        "light_free": LightFreeShader,
    }
    shader = shaders.get(shader_type, "soft_phong")
    #  render by pytorch3d
    renderer_cls = MeshRendererWithDepth
    # renderer_cls = MeshRenderer
    renderer = renderer_cls(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=BlendParams(background_color=(0.0,) * 3),
        ),
    )
    return renderer


def post_process(meshes, rgb, depth, pix_to_face):
    batch_size = len(meshes)
    pix_to_face = pix_to_face.squeeze(
        -1
    ) - meshes.mesh_to_faces_packed_first_idx().view(batch_size, 1, 1)
    pix_to_face[pix_to_face < 0] = -1

    assert not torch.isnan(depth).any()
    assert not torch.isnan(rgb[..., :3]).any(), [
        torch.isnan(rgb[..., i]).sum().item() for i in range(4)
    ]

    assert (
        -1e2 < rgb.min()
        and rgb.max() < 1.5
        and rgb.dtype == depth.dtype == torch.float32
    ), [rgb.min(), rgb.max(), rgb.dtype, depth.dtype]

    # The renderer randomly adds a small value (e.g. 1e-5) uniformly to all pixels.
    # This is enough to significantly change the model output. The line below fixes the issue.
    if not rgb.requires_grad:
        rgb = rgb.mul(255).round().div(255)
    return rgb, pix_to_face
