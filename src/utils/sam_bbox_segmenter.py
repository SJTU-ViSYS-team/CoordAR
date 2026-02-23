from typing import Union, Tuple, Optional, Dict
import numpy as np
import torch
import cv2

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    raise ImportError(
        "Please install Segment Anything. Example: pip install git+https://github.com/facebookresearch/segment-anything.git"
    )


class SegmentAnythingBBox:
    """
    Segment an object from an image using SAM with a bounding box prompt.

    Usage:
      seg = SegmentAnythingBBox(checkpoint_path="/path/to/sam_vit_h_4b8939.pth")
      mask, score = seg.best_mask("/path/to/image.jpg", (x0, y0, x1, y1))
      # mask: np.ndarray[H,W] of bool
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        self.predictor = SamPredictor(sam)

    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            img_bgr = cv2.imread(image, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found: {image}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected HxWx3 image array.")
            return image
        raise TypeError("image must be a file path or an RGB numpy array.")

    def segment(
        self,
        image: Union[str, np.ndarray],
        box: Tuple[int, int, int, int],
        multimask_output: bool = False,
    ) -> Dict[str, np.ndarray]:
        img_rgb = self._load_image(image)
        self.predictor.set_image(img_rgb)
        x0, y0, x1, y1 = map(int, box)
        input_box = np.array([x0, y0, x1, y1])
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            point_coords=None,
            point_labels=None,
            multimask_output=multimask_output,
        )
        return {"masks": masks, "scores": scores, "logits": logits}

    def best_mask(
        self, image: Union[str, np.ndarray], box: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, float]:
        """
        Returns the highest scoring mask for the given box and its score.
        """
        out = self.segment(image, box, multimask_output=True)
        idx = int(np.argmax(out["scores"]))
        return out["masks"][idx], float(out["scores"][idx])


# python -m src.utils.sam_bbox_segmenter
if __name__ == "__main__":
    # Test the SegmentAnythingBBox class
    import matplotlib.pyplot as plt
    import os

    # Example usage - modify these paths as needed
    checkpoint_path = (
        "logs/pretrained/SAM/sam_vit_h_4b8939.pth"  # Download from SAM repo
    )
    test_image_path = "data/BOP/lm/test/000001/rgb/000000.png"  # Your test image
    test_bbox = (308, 190, 376, 255)  # (x0, y0, x1, y1) format

    if not os.path.exists(checkpoint_path):
        print(f"please download SAM checkpoint to {checkpoint_path}")
        print(
            f"wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O {checkpoint_path}"
        )
        exit(1)

    if not os.path.exists(test_image_path):
        print(f"Please provide a test image at {test_image_path}")
        exit(1)

    try:
        # Initialize segmenter
        print("Loading SAM model...")
        segmenter = SegmentAnythingBBox(checkpoint_path)

        # Test segmentation
        print("Running segmentation...")
        mask, score = segmenter.best_mask(test_image_path, test_bbox)
        print(f"Segmentation score: {score:.3f}")
        print(f"Mask shape: {mask.shape}")

        # Visualize results
        img = segmenter._load_image(test_image_path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Bounding box overlay
        axes[1].imshow(img)
        x0, y0, x1, y1 = test_bbox
        rect = plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, color="red", linewidth=2
        )
        axes[1].add_patch(rect)
        axes[1].set_title("Input Bounding Box")
        axes[1].axis("off")

        # Segmentation mask
        axes[2].imshow(img)
        axes[2].imshow(mask, alpha=0.5, cmap="jet")
        axes[2].set_title(f"Segmentation Result (score: {score:.3f})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig("logs/debug/sam_test_result.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Test completed! Result saved as 'sam_test_result.png'")

    except Exception as e:
        print(f"Test failed: {e}")
