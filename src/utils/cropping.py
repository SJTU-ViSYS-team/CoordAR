import math
import torch
import torchvision
import cv2
import numpy as np


def project_points(points_3d, K, TCO):
    assert K.shape[-2:] == (3, 3)
    assert TCO.shape[-2:] == (4, 4)
    batch_size = points_3d.shape[0]
    n_points = points_3d.shape[1]
    device = points_3d.device
    if points_3d.shape[-1] == 3:
        points_3d = torch.cat(
            (points_3d, torch.ones(batch_size, n_points, 1).to(device)), dim=-1
        )
    P = K @ TCO[:, :3]
    suv = (P.unsqueeze(1) @ points_3d.unsqueeze(-1)).squeeze(-1)
    suv = suv / suv[..., [-1]]
    return suv[..., :2]


def project_points_robust(points_3d, K, TCO, z_min=0.1):
    assert K.shape[-2:] == (3, 3)
    assert TCO.shape[-2:] == (4, 4)
    batch_size = points_3d.shape[0]
    n_points = points_3d.shape[1]
    device = points_3d.device
    if points_3d.shape[-1] == 3:
        points_3d = torch.cat(
            (points_3d, torch.ones(batch_size, n_points, 1).to(device)), dim=-1
        )
    P = K @ TCO[:, :3]
    suv = (P.unsqueeze(1) @ points_3d.unsqueeze(-1)).squeeze(-1)
    z = suv[..., -1]
    suv[..., -1] = torch.max(torch.ones_like(z) * z_min, z)
    suv = suv / suv[..., [-1]]
    return suv[..., :2]


def boxes_from_uv(uv):
    assert uv.shape[-1] == 2
    x1 = uv[..., [0]].min(dim=1)[0]
    y1 = uv[..., [1]].min(dim=1)[0]

    x2 = uv[..., [0]].max(dim=1)[0]
    y2 = uv[..., [1]].max(dim=1)[0]

    return torch.cat((x1, y1, x2, y2), dim=1)


def crop_to_aspect_ratio(images, box, masks=None, K=None, depths=None):
    assert images.dim() == 4
    bsz, _, h, w = images.shape
    assert box.dim() == 1
    assert box.shape[0] == 4
    w_output, h_output = box[[2, 3]] - box[[0, 1]]
    boxes = torch.cat(
        (
            torch.arange(bsz).unsqueeze(1).to(box.device).float(),
            box.unsqueeze(0).repeat(bsz, 1).float(),
        ),
        dim=1,
    ).to(images.device)
    images = torchvision.ops.roi_pool(images, boxes, output_size=(h_output, w_output))
    if masks is not None:
        assert masks.dim() == 4
        masks = torchvision.ops.roi_pool(masks, boxes, output_size=(h_output, w_output))
    if depths is not None:
        assert depths.dim() == 4, depths.shape
        depths = torchvision.ops.roi_pool(
            depths, boxes, output_size=(h_output, w_output)
        )
    if K is not None:
        assert K.dim() == 3
        assert K.shape[0] == bsz
        K = get_K_crop_resize(
            K, boxes[:, 1:], orig_size=(h, w), crop_resize=(h_output, w_output)
        )
    return images, masks, K, depths


def get_K_crop_resize(K, boxes, orig_size, crop_resize):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    if isinstance(K, np.ndarray):
        K = torch.as_tensor(K)
    if isinstance(boxes, np.ndarray):
        boxes = torch.as_tensor(boxes)
    if len(K.shape) == 2:
        K = K[None, ...]
    if len(boxes.shape) == 1:
        boxes = boxes[None, ...]
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4,)
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    orig_size = torch.as_tensor(orig_size, dtype=torch.float).clone()
    crop_resize = torch.as_tensor(crop_resize, dtype=torch.float).clone()

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K


def get_K_crop_resize_v2(K, boxes, orig_size, crop_resize):
    """
    simplified from v1, well tested
    """
    if isinstance(K, np.ndarray):
        K = torch.as_tensor(K)
    if isinstance(boxes, np.ndarray):
        boxes = torch.as_tensor(boxes)
    if len(K.shape) == 2:
        K = K[None, ...]
    if len(boxes.shape) == 1:
        boxes = boxes[None, ...]
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4,)
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    orig_size = torch.as_tensor(orig_size, dtype=torch.float).clone()
    crop_resize = torch.as_tensor(crop_resize, dtype=torch.float).clone()

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # # Resize (upsample)
    orig_cx_diff = K[:, 0, 2] - crop_cj
    orig_cy_diff = K[:, 1, 2] - crop_ci
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height

    print("scale_xy", scale_x, scale_y)
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy

    print("new_k", new_K)
    return new_K


def crop_boxes(bsz, orig_size, crop_size, points, K, TCO):
    assert points.shape[0] == (bsz) and points.shape[2] == 3
    assert K.shape == (bsz, 3, 3)
    assert TCO.shape == (bsz, 4, 4)
    uv = project_points(points, K, TCO)
    boxes_rend = boxes_from_uv(uv)
    boxes_crop = deepim_crops_robust(
        obs_boxes=boxes_rend,
        K=K,
        TCO_pred=TCO,
        O_vertices=points,
        output_size=crop_size,
        lamb=1.4,
    )

    K_crop = get_K_crop_resize(
        K=K.clone(), boxes=boxes_crop, orig_size=orig_size, crop_resize=crop_size
    )

    return K_crop.detach(), boxes_rend, boxes_crop


def crop_images(boxes_crop, output_size, images, masks, depth):
    batch_size = boxes_crop.shape[0]
    device = boxes_crop.device
    boxes = torch.cat(
        (torch.arange(batch_size).unsqueeze(1).to(device).float(), boxes_crop), dim=1
    )
    images_cropped = torchvision.ops.roi_align(
        images, boxes, output_size=output_size, sampling_ratio=4
    )
    # print(images.shape)
    # cv2.imwrite("original_rgb.png", images[0].permute(1,2,0).cpu().numpy().astype(np.uint8))
    # cv2.imwrite("cropped_rgb.png", images_cropped[0].permute(1,2,0).cpu().numpy().astype(np.uint8))
    masks_cropped = (
        None
        if masks is None
        else torchvision.ops.roi_align(
            masks.float(), boxes, output_size=output_size, sampling_ratio=4
        ).bool()
    )
    depth_cropped = (
        None
        if depth is None
        else torchvision.ops.roi_align(
            depth.float(), boxes, output_size=output_size, sampling_ratio=4
        )
    )
    return images_cropped, masks_cropped, depth_cropped


def deepim_boxes(
    rend_center_uv, obs_boxes, rend_boxes, lamb=1.4, im_size=(240, 320), clamp=False
):
    """
    gt_boxes: N x 4
    crop_boxes: N x 4
    """
    lobs, robs, uobs, dobs = obs_boxes[:, [0, 2, 1, 3]].t()
    lrend, rrend, urend, drend = rend_boxes[:, [0, 2, 1, 3]].t()
    xc = rend_center_uv[..., 0, 0]
    yc = rend_center_uv[..., 0, 1]
    lobs, robs = lobs.unsqueeze(-1), robs.unsqueeze(-1)
    uobs, dobs = uobs.unsqueeze(-1), dobs.unsqueeze(-1)
    lrend, rrend = lrend.unsqueeze(-1), rrend.unsqueeze(-1)
    urend, drend = urend.unsqueeze(-1), drend.unsqueeze(-1)

    xc, yc = xc.unsqueeze(-1), yc.unsqueeze(-1)
    w = max(im_size)
    h = min(im_size)
    r = w / h

    xdists = torch.cat(
        ((lobs - xc).abs(), (lrend - xc).abs(), (robs - xc).abs(), (rrend - xc).abs()),
        dim=1,
    )
    ydists = torch.cat(
        ((uobs - yc).abs(), (urend - yc).abs(), (dobs - yc).abs(), (drend - yc).abs()),
        dim=1,
    )
    xdist = xdists.max(dim=1)[0]
    ydist = ydists.max(dim=1)[0]
    width = torch.max(xdist, ydist * r) * 2 * lamb
    height = torch.max(xdist / r, ydist) * 2 * lamb

    xc, yc = xc.squeeze(-1), yc.squeeze(-1)
    x1, y1, x2, y2 = xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2
    boxes = torch.cat(
        (x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1
    )
    assert not clamp
    if clamp:
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, h - 1)
    return boxes


def deepim_crops_robust(obs_boxes, output_size, K, TCO_pred, O_vertices, lamb=1.4):
    batch_size, _ = obs_boxes.shape
    device = obs_boxes.device
    uv = project_points_robust(O_vertices, K, TCO_pred)
    rend_boxes = boxes_from_uv(uv)
    rend_center_uv = project_points_robust(
        torch.zeros(batch_size, 1, 3).to(device), K, TCO_pred
    )
    boxes = deepim_boxes(
        rend_center_uv, obs_boxes, rend_boxes, im_size=output_size, lamb=lamb
    )
    return boxes


def bbox_from_mask(instance_mask, max_window=640, interval=40):
    """Compute square image crop window."""
    img_length, img_width = instance_mask.shape
    pts = np.array(list(zip(*instance_mask.nonzero())))
    y1, x1, y2, x2 = *np.min(pts, axis=0), *np.max(pts, axis=0)

    window_size = (max(y2 - y1, x2 - x1) // interval + 1) * interval
    window_size = min(window_size, max_window)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_length:
        delt = rmax - img_length
        rmax = img_length
        rmin -= delt
    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    # xyxy
    return np.array([cmin, rmin, cmax, rmax], dtype=np.float32)


def bbox_from_uvd(center, crop_size, d0=400):
    u, v, d = center
    alpha = d / d0
    rmin = int(v - crop_size / 2 / alpha)
    cmin = int(u - crop_size / 2 / alpha)
    rmax = int(v + crop_size / 2 / alpha)
    cmax = int(u + crop_size / 2 / alpha)
    return rmin, rmax, cmin, cmax


def square_bbox_from_detection(bbox_xyxy, image_size):
    img_w, img_h = image_size

    new_bbox = np.zeros_like(bbox_xyxy)
    # clip
    new_bbox[0] = max(0, bbox_xyxy[0])
    new_bbox[1] = max(0, bbox_xyxy[1])
    new_bbox[2] = min(image_size[0], bbox_xyxy[2])
    new_bbox[3] = min(image_size[1], bbox_xyxy[3])

    bbox_xywh = xyxy2xywh(new_bbox)
    w, h = bbox_xywh[2:]
    x1, y1, x2, y2 = new_bbox
    center_x, center_y = x1 + w / 2, y1 + h / 2
    h = y2 - y1
    w = x2 - x1
    s = max(h, w)
    s = max(2, s)
    if center_x + s / 2 > img_w:
        center_x = max(0, center_x - img_w - (center_x + s / 2))
    if center_y + s / 2 > img_h:
        center_y = max(0, center_y - img_h - (center_y + s / 2))

    new_bbox[0] = max(0, int(center_x - s / 2))
    new_bbox[1] = max(0, int(center_y - s / 2))
    new_bbox[2] = min(int(center_x + s / 2), img_w)
    new_bbox[3] = min(int(center_y + s / 2), img_h)
    assert new_bbox[2] > new_bbox[0]
    assert new_bbox[3] > new_bbox[1]
    return new_bbox


def square_bbox_from_detection_batch(bbox_xyxy, image_size):
    square_box = []
    for i in range(bbox_xyxy.shape[0]):
        square_box.append(square_bbox_from_detection(bbox_xyxy[i], image_size))
    return np.stack(square_box)


def xywh2xyxy(bbox_xywh):
    bbox_xyxy = bbox_xywh.copy()
    bbox_xyxy[..., 2] += bbox_xywh[..., 0]
    bbox_xyxy[..., 3] += bbox_xywh[..., 1]
    return bbox_xyxy


def xyxy2xywh(bbox_xyxy):
    bbox_xywh = bbox_xyxy.copy()
    bbox_xywh[..., 2] -= bbox_xyxy[..., 0]
    bbox_xywh[..., 3] -= bbox_xyxy[..., 1]
    return bbox_xywh


def bbox_crop(image, bbox, bbox_format="xywh"):
    # bbox: [x, y, w, h]

    # 计算bbox在图像中的位置
    x, y, w, h = 0, 0, 0, 0
    x1, y1, x2, y2 = 0, 0, 0, 0
    if bbox_format == "xywh":
        x, y, w, h = bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x + w, image.shape[1]), min(y + h, image.shape[0])
    elif bbox_format == "xyxy":
        x, y, x2, y2 = bbox
        w, h = x2 - x, y2 - y
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])

    # 对图像进行bbox crop
    cropped_image = image[y1:y2, x1:x2]

    # 将超出bbox的区域设置为零
    if x < 0:
        cropped_image = np.concatenate(
            [
                np.zeros(
                    (cropped_image.shape[0], -x, cropped_image.shape[2]),
                    dtype=cropped_image.dtype,
                ),
                cropped_image,
            ],
            axis=1,
        )
    if y < 0:
        cropped_image = np.concatenate(
            [
                np.zeros(
                    (-y, cropped_image.shape[1], cropped_image.shape[2]),
                    dtype=cropped_image.dtype,
                ),
                cropped_image,
            ],
            axis=0,
        )
    if x + w > image.shape[1]:
        cropped_image = np.concatenate(
            [
                cropped_image,
                np.zeros(
                    (
                        cropped_image.shape[0],
                        x + w - image.shape[1],
                        cropped_image.shape[2],
                    ),
                    dtype=cropped_image.dtype,
                ),
            ],
            axis=1,
        )
    if y + h > image.shape[0]:
        cropped_image = np.concatenate(
            [
                cropped_image,
                np.zeros(
                    (
                        y + h - image.shape[0],
                        cropped_image.shape[1],
                        cropped_image.shape[2],
                    ),
                    dtype=cropped_image.dtype,
                ),
            ],
            axis=0,
        )

    return cropped_image


def crop_xyxy(image, box_xyxy):
    return image[box_xyxy[1] : box_xyxy[3], box_xyxy[0] : box_xyxy[2]]


def keypoints_rel2abs(keypoints, bbox, crop_size, size_orig):
    bbox = xyxy2xywh(bbox)
    bbox_x, bbox_y, bbox_width, bbox_height = bbox
    orig_width, orig_height = size_orig
    scale_x = bbox_width / crop_size[0]
    scale_y = bbox_height / crop_size[1]
    abs_keypoints = []
    for rel_keypoint in keypoints:
        rel_x, rel_y = rel_keypoint
        abs_x = (rel_x * scale_x) + bbox_x
        abs_y = (rel_y * scale_y) + bbox_y
        abs_keypoints.append((abs_x, abs_y))

    return np.stack(abs_keypoints)


# ---------------------------------------------------------------------------------------
# GDRNet cropping utils
def crop_resize_by_warp_affine(
    img, center, scale, output_size, rot=0.0, interpolation=cv2.INTER_LINEAR
):
    """
    output_size: int or (w, h)
    rot: angle in deg, negative value means clockwise rotation
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(
        img,
        trans,
        (int(output_size[0]), int(output_size[1])),
        flags=interpolation,
    )

    return dst_img


def crop_resize_by_warp_affine_batch(
    imgs, centers, scales, output_size, rot=0.0, interpolation=cv2.INTER_LINEAR
):
    bs = imgs.shape[0]
    dst_imgs = []
    for i in range(bs):
        dst_imgs.append(
            crop_resize_by_warp_affine(
                imgs[i], centers[i], scales[i], output_size, rot, interpolation
            )
        )
    return np.stack(dst_imgs)


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=np.array([0, 0], dtype=np.float32),
    inv=False,
):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


# ---------------------------------------------------------------------------------------


# 获得与平面内旋转近似相等的渲染参数变换
def adjoint_render_param(inplane_rot, K, TCO, w, h, bbox, out_res):
    # inplane_rot： 角度
    # K: 相机内参, 整图

    # 物体中心的投影
    center = TCO[:3, 3]
    d = center[2]
    center_2d = center @ K.T / d

    inplane_rot = math.radians(inplane_rot)
    rot2d = torch.tensor(
        [
            [math.cos(inplane_rot), -math.sin(inplane_rot), 0],
            [math.sin(inplane_rot), math.cos(inplane_rot), 0],
            [0, 0, 1],
        ]
    ).float()

    rot_center = center @ rot2d.T
    rot_center_2d = rot_center @ K.T / d

    # K_crop对应的中心绕光心旋转若干度,平移bbox中心到目标
    bbox = bbox.clone()
    bbox[[0, 2]] = bbox[[0, 2]] + (rot_center_2d - center_2d)[0]
    bbox[[1, 3]] = bbox[[1, 3]] + (rot_center_2d - center_2d)[1]
    K_crop = get_K_crop_resize(K, bbox, orig_size=(h, w), crop_resize=out_res)[0]
    # 位姿中旋转和平移发生改变
    # 旋转已经算过了
    TCO_adj = TCO.clone()
    TCO_adj[:3, 3] = rot_center

    return TCO_adj, K_crop


def bbox_from_pose(TCO, K, model_pts, h, w):
    """
    Args:
        pose: 4x4
        K: 3x3
        model_pts: Nx3
        diameter: float
    Returns:
        (4,) bbox in xywh format
    """
    model_pts = model_pts @ TCO[:3, :3].T + TCO[:3, 3]
    model_pts = model_pts @ K.T
    model_pts = model_pts / model_pts[:, 2][:, None]
    x = model_pts[:, 0].clip(0, w - 1)
    y = model_pts[:, 1].clip(0, h - 1)
    x1 = x.min()
    x2 = x.max()
    y1 = y.min()
    y2 = y.max()
    return np.array([x1, y1, x2 - x1, y2 - y1])


def encode_scale_invariant_trans(
    z, gt_center_2d, detection_center_2d, resize_ratio, bw, bh, z_scale, fov
):
    # if resize_ratio > 1.0 , the object is zoomed in, so the z should be scaled down
    # if images of two similar objects has same resize_ratio,
    # the real z is inversely proportional to fov
    z_ratio = z / resize_ratio * z_scale * np.tan(np.deg2rad(fov) / 2)
    obj_center = gt_center_2d
    delta_c = obj_center - detection_center_2d
    gt_t_param = np.array([delta_c[0] / bw, delta_c[1] / bh, z_ratio], dtype=np.float32)
    return gt_t_param
