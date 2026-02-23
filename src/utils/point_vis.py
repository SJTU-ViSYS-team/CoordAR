import io
from typing import List, Union, Optional
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


def show_multiple_point_clouds_plotly(
    points_list: List[np.ndarray],
    colors_list: Optional[List[str]] = None,
    labels_list: Optional[List[str]] = None,
    title: str = "Multiple Point Clouds",
    size: Union[int, List[int]] = 3,
    save_path: Optional[str] = None,
    show: bool = True,
    fig: Optional[go.Figure] = None,
) -> go.Figure:
    """
    Visualize multiple point clouds with different colors and legends using Plotly.

    Args:
        points_list: List of point clouds, each as (N, 3) numpy array
        colors_list: List of colors for each point cloud (hex strings, color names, or RGB)
        labels_list: List of labels for legend (if None, uses "Cloud 0", "Cloud 1", etc.)
        title: Plot title
        size: Point size (single value or list for each cloud)
        save_path: Path to save HTML file (optional)
        show: Whether to display the plot
        fig: Existing figure to add to (optional)

    Returns:
        Plotly figure object
    """
    if fig is None:
        fig = go.Figure()

    # Default colors if not provided
    if colors_list is None:
        default_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        colors_list = [
            default_colors[i % len(default_colors)] for i in range(len(points_list))
        ]

    # Default labels if not provided
    if labels_list is None:
        labels_list = [f"Cloud {i}" for i in range(len(points_list))]

    # Handle size parameter
    if isinstance(size, int):
        size_list = [size] * len(points_list)
    else:
        size_list = size

    # Add each point cloud as a separate trace
    all_points = []
    for i, points in enumerate(points_list):
        if len(points) == 0:
            continue

        all_points.append(points)

        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=size_list[i] if i < len(size_list) else size_list[0],
                    color=colors_list[i] if i < len(colors_list) else colors_list[0],
                ),
                name=labels_list[i] if i < len(labels_list) else f"Cloud {i}",
                showlegend=True,
            )
        )

    # Calculate unified axis ranges for all point clouds
    if all_points:
        combined_points = np.vstack(all_points)
        x_min, y_min, z_min = combined_points.min(axis=0)
        x_max, y_max, z_max = combined_points.max(axis=0)
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

        # Add some padding
        padding = max_range * 0.1
        center_x, center_y, center_z = (
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2,
        )
        range_half = (max_range + padding) / 2

        fig.update_layout(
            title=title,
            scene=dict(
                aspectmode="cube",
                xaxis=dict(range=[center_x - range_half, center_x + range_half]),
                yaxis=dict(range=[center_y - range_half, center_y + range_half]),
                zaxis=dict(range=[center_z - range_half, center_z + range_half]),
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
            ),
        )

    if save_path is not None:
        fig.write_html(save_path)
    if show:
        fig.show()

    return fig


def test_show_multiple_point_clouds_plotly():
    """
    Test function for show_multiple_point_clouds_plotly with various scenarios.
    """
    print("Testing show_multiple_point_clouds_plotly function...")

    # Generate sample point clouds
    np.random.seed(42)

    # Test 1: Basic usage with 3 point clouds
    print("Test 1: Basic usage with 3 point clouds")
    cloud1 = np.random.randn(100, 3) * 0.5  # Small cloud at origin
    cloud2 = np.random.randn(80, 3) * 0.3 + np.array([2, 0, 0])  # Cloud shifted in X
    cloud3 = np.random.randn(120, 3) * 0.4 + np.array(
        [0, 2, 1]
    )  # Cloud shifted in Y and Z

    points_list = [cloud1, cloud2, cloud3]
    colors_list = ["red", "blue", "green"]
    labels_list = ["Origin Cloud", "X-shifted Cloud", "Y-Z shifted Cloud"]

    fig1 = show_multiple_point_clouds_plotly(
        points_list=points_list,
        colors_list=colors_list,
        labels_list=labels_list,
        title="Test 1: Basic Multiple Point Clouds",
        show=False,
    )
    print("✓ Test 1 completed")

    # Test 2: Using default colors and labels
    print("Test 2: Using default colors and labels")
    fig2 = show_multiple_point_clouds_plotly(
        points_list=points_list, title="Test 2: Default Colors and Labels", show=False
    )
    print("✓ Test 2 completed")

    # Test 3: Different point sizes
    print("Test 3: Different point sizes for each cloud")
    fig3 = show_multiple_point_clouds_plotly(
        points_list=points_list,
        colors_list=["#FF6B6B", "#4ECDC4", "#45B7D1"],
        labels_list=["Small Points", "Medium Points", "Large Points"],
        size=[2, 4, 6],
        title="Test 3: Different Point Sizes",
        show=False,
    )
    print("✓ Test 3 completed")

    # Test 4: Many point clouds (testing color cycling)
    print("Test 4: Many point clouds (color cycling)")
    many_clouds = []
    for i in range(8):
        angle = i * 2 * np.pi / 8
        center = np.array([np.cos(angle) * 3, np.sin(angle) * 3, i * 0.5])
        cloud = np.random.randn(50, 3) * 0.2 + center
        many_clouds.append(cloud)

    fig4 = show_multiple_point_clouds_plotly(
        points_list=many_clouds,
        title="Test 4: Eight Point Clouds in Circle",
        show=False,
    )
    print("✓ Test 4 completed")

    # Test 5: Geometric shapes
    print("Test 5: Geometric shapes")
    # Create a sphere
    u = np.random.uniform(0, 2 * np.pi, 200)
    v = np.random.uniform(0, np.pi, 200)
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)
    sphere = np.column_stack([x_sphere, y_sphere, z_sphere])

    # Create a cube
    cube_points = []
    for i in range(300):
        face = np.random.randint(0, 6)
        if face == 0:  # x = 1
            point = [1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        elif face == 1:  # x = -1
            point = [-1, np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
        elif face == 2:  # y = 1
            point = [np.random.uniform(-1, 1), 1, np.random.uniform(-1, 1)]
        elif face == 3:  # y = -1
            point = [np.random.uniform(-1, 1), -1, np.random.uniform(-1, 1)]
        elif face == 4:  # z = 1
            point = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 1]
        else:  # z = -1
            point = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), -1]
        cube_points.append(point)
    cube = np.array(cube_points) + np.array([3, 0, 0])

    # Create a line
    t = np.linspace(0, 4 * np.pi, 150)
    line = np.column_stack([t * 0.1, np.sin(t) * 0.5, np.cos(t) * 0.5]) + np.array(
        [-3, 0, 0]
    )

    geometric_shapes = [sphere, cube, line]
    shape_colors = ["purple", "orange", "cyan"]
    shape_labels = ["Sphere", "Cube", "Helix"]

    fig5 = show_multiple_point_clouds_plotly(
        points_list=geometric_shapes,
        colors_list=shape_colors,
        labels_list=shape_labels,
        title="Test 5: Geometric Shapes",
        size=[3, 4, 2],
        show=False,
    )
    print("✓ Test 5 completed")

    # Test 6: Empty cloud handling
    print("Test 6: Handling empty clouds")
    clouds_with_empty = [cloud1, np.array([]).reshape(0, 3), cloud2]
    fig6 = show_multiple_point_clouds_plotly(
        points_list=clouds_with_empty,
        colors_list=["red", "blue", "green"],
        labels_list=["Cloud 1", "Empty Cloud", "Cloud 2"],
        title="Test 6: With Empty Cloud",
        show=False,
    )
    print("✓ Test 6 completed")

    # Show one of the figures as an example
    print("\nShowing Test 5 (Geometric Shapes) as example...")
    fig5.show()

    print("\nAll tests completed successfully!")
    print("Functions tested:")
    print("- Basic multiple point cloud visualization")
    print("- Default color and label handling")
    print("- Different point sizes")
    print("- Color cycling for many clouds")
    print("- Geometric shape visualization")
    print("- Empty cloud handling")

    return {
        "basic": fig1,
        "default": fig2,
        "sizes": fig3,
        "many_clouds": fig4,
        "geometric": fig5,
        "empty_handling": fig6,
    }


# python -m src.utils.point_vis
if __name__ == "__main__":
    # Run tests when script is executed directly
    test_results = test_show_multiple_point_clouds_plotly()
