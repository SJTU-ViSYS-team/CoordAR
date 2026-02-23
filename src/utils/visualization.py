import io
from typing import List, Union
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import torch


def plot_to_image(figure, dpi=100):
    """Converts matplotlib fig to a png for logging with tf.summary.image."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="raw", dpi=dpi)
    plt.close(figure)
    buffer.seek(0)
    image = np.reshape(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8),
        newshape=(int(figure.bbox.bounds[3]), int(figure.bbox.bounds[2]), -1),
    )
    return image[..., :3]


def combine_point_clouds(
    points_list: List[np.ndarray],
    colors_list: List[Union[str, np.ndarray, list]] = None,
    default_color: str = "#1f77b4",  # 默认颜色（Plotly蓝色）
) -> tuple:
    """
    合并多个点云并分配颜色，返回合并后的点云和颜色数组

    Args:
        points_list: 点云列表，每个元素是 (N, 3) 的 numpy 数组
        colors_list: 颜色列表，支持以下格式：
            - None: 所有点使用 default_color
            - 字符串: 如 'red' (所有点同色)
            - 数组/列表: 为每个点云单独指定颜色（长度需与 points_list 一致）
        default_color: 未指定颜色时的默认值

    Returns:
        combined_points: (M, 3) 合并后的点云
        combined_colors: (M,) 合并后的颜色数组（HEX字符串或数值）
    """
    # 合并所有点云
    combined_points = np.vstack(points_list)

    # 处理颜色
    if colors_list is None:
        # 所有点使用默认颜色
        combined_colors = np.full(len(combined_points), default_color)
    elif isinstance(colors_list, str):
        # 所有点使用单一颜色
        combined_colors = np.full(len(combined_points), colors_list)
    else:
        # 为每个点云分配颜色
        combined_colors = []
        for i, points in enumerate(points_list):
            if i < len(colors_list):
                color = colors_list[i]
                if isinstance(color, (str, np.str_)):
                    # 单色：重复填充
                    combined_colors.extend([color] * len(points))
                else:
                    # 每个点单独颜色（如渐变数组）
                    combined_colors.extend(color)
            else:
                # 超出颜色列表的点云用默认颜色
                combined_colors.extend([default_color] * len(points))
        combined_colors = np.array(combined_colors)

    return combined_points, combined_colors


def show_point_cloud_plotly(
    points, colors=None, title="", size=3, save_path=None, show=True, fig=None
):

    if fig is None:
        fig = go.Figure()
    assert points.shape[1] == 3, "Points should be of shape (N, 3)"
    if colors is None:
        assert colors.shape[0] == points.shape[0]
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=size, color=colors),
            name="points",
        )
    )
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="cube",  # 强制XYZ等比例
            xaxis=dict(range=[x_min, x_min + max_range]),  # 统一使用最大范围
            yaxis=dict(range=[y_min, y_min + max_range]),
            zaxis=dict(range=[z_min, z_min + max_range]),
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
    )
    if save_path is not None:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def show_point_cloud_normal_plotly(
    points, normals, title="", sizeref=1, save_path=None, show=True
):

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=3, color="blue"),
            name="points",
        )
    )
    fig.add_trace(
        go.Cone(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            u=normals[:, 0],
            v=normals[:, 1],
            w=normals[:, 2],
            colorscale="Blues",
            sizemode="absolute",
            sizeref=sizeref,
            showscale=False,
            name="normals",
        )
    )
    fig.update_layout(title=title)
    if save_path is not None:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def show_rays_plotly(
    points, rays, length=1000.0, title="Rays Visualization", save_path=None, show=True
):
    """
    Visualizes rays in 3D space using Plotly.

    Parameters:
    - points: ndarray of shape (N, 3), where each row represents a starting point of a ray (x, y, z).
    - rays: ndarray of shape (N, 3), where each row represents a ray direction (dx, dy, dz).
    - title: str, title of the plot.
    """
    # Validate input dimensions
    if points.shape != rays.shape or points.shape[1] != 3:
        raise ValueError("Both 'points' and 'rays' must have shape (N, 3).")

    scaled_rays = -rays * length

    # Extract components
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    u, v, w = scaled_rays[:, 0], scaled_rays[:, 1], scaled_rays[:, 2]

    # Create the figure
    fig = go.Figure()

    # Add rays as quiver plots
    for i in range(len(points)):
        fig.add_trace(
            go.Scatter3d(
                x=[x[i], x[i] + u[i]],
                y=[y[i], y[i] + v[i]],
                z=[z[i], z[i] + w[i]],
                mode="lines+markers",
                marker=dict(size=3, color="red"),
                line=dict(color="blue", width=2),
                name=f"Ray {i}",
            )
        )

    # Set layout
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        showlegend=False,
    )

    if save_path is not None:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def normalize_single_channel(conf):
    return (conf - conf.min()) / (conf.max() - conf.min())


def show_points(scene_points, colors=None, save_path=None, show=True):
    sizes = np.ones(scene_points.shape[0]) * 5
    opacity = 1
    max_size = scene_points.max()
    min_size = scene_points.min()

    if colors is None:
        colors = np.ones(scene_points.shape[0])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=scene_points[:, 0],
                y=scene_points[:, 1],
                z=scene_points[:, 2],
                mode="markers",
                name="scene",
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=opacity,
                ),
            ),
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[min_size, max_size],
            ),
            yaxis=dict(
                range=[min_size, max_size],
            ),
            zaxis=dict(
                range=[min_size, max_size],
            ),
        ),
    )
    if save_path is not None:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def draw_registration_plotly(
    source,
    target,
    src_feats,
    tgt_feats,
    transformation,
    correspondences,
    save_path=None,
    show=True,
    fig=None,
):
    # source: M, 3, tensor
    # target: N, 3
    # src_feats: M, D
    # tgt_feats: N, D
    # transformation: 4, 4
    # correspondences: K, 2
    if fig is None:
        fig = go.Figure()
    M = source.shape[0]
    joint_feats = torch.cat([src_feats, tgt_feats], dim=0)
    joint_feats_pca = (torch_pca(joint_feats.unsqueeze(0)).squeeze(0) + 0.5).clip(0, 1)
    src_colors = joint_feats_pca[:M].cpu().numpy()
    tgt_colors = joint_feats_pca[M:].cpu().numpy()

    # transform
    source_trans = source @ transformation[:3, :3].T + transformation[:3, 3]
    source = source.cpu().numpy()
    source_trans = source_trans.cpu().numpy()
    target = target.cpu().numpy()
    correspondences = correspondences.cpu().numpy()

    src_scatter = go.Scatter3d(
        x=source[:, 0],
        y=source[:, 1],
        z=source[:, 2],
        mode="markers",
        marker=dict(size=5, color=src_colors),
        name="Source",
    )
    target_scatter = go.Scatter3d(
        x=target[:, 0],
        y=target[:, 1],
        z=target[:, 2],
        mode="markers",
        marker=dict(size=5, color=tgt_colors),
        name="Target",
    )
    src_trans_scatter = go.Scatter3d(
        x=source_trans[:, 0],
        y=source_trans[:, 1],
        z=source_trans[:, 2],
        mode="markers",
        marker=dict(size=5, color=src_colors),
        name="Source_trans",
    )
    # lines
    lines = []
    for idx1, idx2 in correspondences:
        lines.append(
            go.Scatter3d(
                x=[source[idx1, 0], target[idx2, 0]],
                y=[source[idx1, 1], target[idx2, 1]],
                z=[source[idx1, 2], target[idx2, 2]],
                mode="lines",
                line=dict(color="green", width=2),
                showlegend=False,
            )
        )

    fig = go.Figure()
    fig.add_trace(src_scatter)
    fig.add_trace(target_scatter)
    fig.add_trace(src_trans_scatter)
    for line in lines:
        fig.add_trace(line)
    fig.update_layout(
        title="3D Point Correspondence",
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        legend=dict(x=0.8, y=0.9),
    )
    if save_path is not None:
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig
