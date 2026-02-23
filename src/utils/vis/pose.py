import cv2
import ipdb
import mmcv
import numpy as np

from src.utils.vis.colormap import colormap


def draw_projected_box3d(
    image,
    qs,
    color=(255, 0, 255),
    middle_color=None,
    bottom_color=None,
    thickness=2,
):
    """Draw 3d bounding box in image
    qs: (8,2), projected 3d points array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    # Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py
    qs = qs.astype(np.int32)
    color = mmcv.color_val(color)  # top color
    colors = colormap(rgb=False, maximum=255)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        # use LINE_AA for opencv3
        # CV_AA for opencv2?
        # bottom: blue
        i, j = k + 4, (k + 1) % 4 + 4
        if bottom_color is None:
            _bottom_color = tuple(int(_c) for _c in colors[k % len(colors)])
        else:
            _bottom_color = tuple(int(_c) for _c in mmcv.color_val(bottom_color))
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            _bottom_color,
            thickness,
            cv2.LINE_AA,
        )

        # middle: colormap
        i, j = k, k + 4
        if middle_color is None:
            _middle_color = tuple(int(_c) for _c in colors[k % len(colors)])
        else:
            _middle_color = tuple(int(_c) for _c in mmcv.color_val(middle_color))
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            _middle_color,
            thickness,
            cv2.LINE_AA,
        )

        # top: pink/red
        i, j = k, (k + 1) % 4
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            color,
            thickness,
            cv2.LINE_AA,
        )

    # # method 2
    # draw pillars in blue color-------------------
    # for i, j in zip(range(4), range(4, 8)):
    #     image = cv2.line(image, tuple(qs[i]), tuple(qs[j]), (255), thickness)

    # # draw bottom layer in red color
    # image = cv2.drawContours(image, [qs[4:]], -1, (0, 0, 255), thickness)
    # # draw top layer in red color
    # image = cv2.drawContours(image, [qs[:4]], -1, (0, 255, 0), thickness)
    # ---------------------------
    return image


def draw_projected_box3d_xyz(
    image,
    qs,
    color=(200, 200, 200),
    thickness=2,
):
    """Draw 3d bounding box in image
    qs: (8,2), projected 3d points array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    qs = qs.astype(np.int32)
    normal_edge_color = mmcv.color_val(color)  # top color
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    axis_color = {
        (6, 7): (255, 0, 0),  # z-axis in blue
        (5, 6): (0, 255, 0),  # y-axis in green
        (2, 6): (0, 0, 255),  # x-axis in red
    }
    for edge in edges:
        i, j = edge
        if edge in axis_color:
            edge_color = axis_color[edge]
        else:
            edge_color = normal_edge_color
        cv2.line(
            image,
            (qs[i, 0], qs[i, 1]),
            (qs[j, 0], qs[j, 1]),
            edge_color,
            thickness,
            cv2.LINE_AA,
        )
    return image


def get_bbox3d_and_center(pts):
    """
    pts: Nx3
    ---
    bb: bb3d+center, 9x3
    """
    bb = []
    minx, maxx = min(pts[:, 0]), max(pts[:, 0])
    miny, maxy = min(pts[:, 1]), max(pts[:, 1])
    minz, maxz = min(pts[:, 2]), max(pts[:, 2])
    avgx = np.average(pts[:, 0])
    avgy = np.average(pts[:, 1])
    avgz = np.average(pts[:, 2])
    # (000)-->
    # bb.append([minx, miny, minz])
    # bb.append([minx, maxy, minz])
    # bb.append([minx, miny, maxz])
    # bb.append([minx, maxy, maxz])
    # bb.append([maxx, miny, minz])
    # bb.append([maxx, maxy, minz])
    # bb.append([maxx, miny, maxz])
    # bb.append([maxx, maxy, maxz])
    # bb.append([avgx, avgy, avgz])
    # bb = np.asarray(bb, dtype=np.float32)
    # NOTE: we use a different order from roi10d
    """
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    bb = np.array(
        [
            [maxx, maxy, maxz],
            [minx, maxy, maxz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, maxy, minz],
            [minx, maxy, minz],
            [minx, miny, minz],
            [maxx, miny, minz],
            [avgx, avgy, avgz],
        ],
        dtype=np.float32,
    )
    return bb


def get_bbox3d_from_center_ext(center, extent):
    """
    center: 3
    extent: 3
    """
    x, y, z = center
    dx, dy, dz = extent

    return np.array(
        [
            [x + dx / 2, y + dy / 2, z + dz / 2],
            [x - dx / 2, y + dy / 2, z + dz / 2],
            [x - dx / 2, y - dy / 2, z + dz / 2],
            [x + dx / 2, y - dy / 2, z + dz / 2],
            [x + dx / 2, y + dy / 2, z - dz / 2],
            [x - dx / 2, y + dy / 2, z - dz / 2],
            [x - dx / 2, y - dy / 2, z - dz / 2],
            [x + dx / 2, y - dy / 2, z - dz / 2],
            [x, y, z],
        ],
        dtype=np.float32,
    )


def points_to_2D(points, R, T, K):
    """
    discription: project 3D points to 2D image plane

    :param points: (N, 3)
    :param R: (3, 3)
    :param T: (3, )
    :param K: (3, 3)
    :return: points_2D: (N, 2), z: (N,)
    """
    # using opencv
    # cv2Proj2d = cv2.projectPoints(model_points, R_vec, T, K, None)[0]
    points_in_world = np.matmul(R, points.T) + T.reshape((3, 1))  # (3, N)
    points_in_camera = np.matmul(
        K, points_in_world
    )  # (3, N) # z is not changed in this step
    N = points_in_world.shape[1]
    points_2D = np.zeros((2, N))
    points_2D[0, :] = points_in_camera[0, :] / (points_in_camera[2, :] + 1e-15)
    points_2D[1, :] = points_in_camera[1, :] / (points_in_camera[2, :] + 1e-15)
    z = points_in_world[2, :]
    return points_2D.T, z
