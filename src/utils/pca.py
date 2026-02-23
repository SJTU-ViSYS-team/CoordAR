import ipdb
import torch


def torch_pca(features: torch.Tensor, mask=None, n_components: int = 3):
    """
    Perform PCA using PyTorch.

    Args:
        features (torch.Tensor): Input features of shape (b, n, d).
        n_components (int): Number of principal components to keep.

    Returns:
        torch.Tensor: Projected features of shape (b, n, n_components).
    """
    # Ensure mask is properly shaped
    if mask is None:
        mask = torch.ones_like(features[..., 0])  # (b, n)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)  # (b, n, 1)
    # Centering the features
    mean = torch.mean(features, dim=1, keepdim=True)

    # Compute weighted mean
    valid_counts = mask.sum(dim=1, keepdim=True)  # (b, 1, d)
    mean = (features * mask).sum(dim=1, keepdim=True) / valid_counts

    centered_features = (features - mean) * mask

    # Compute covariance matrix (C = X^T * X / (n-1))
    cov_matrix = torch.bmm(centered_features.permute(0, 2, 1), centered_features) / (
        valid_counts - 1
    )

    # Eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_idx = torch.argsort(eigvals, descending=True, dim=1)
    eigvecs = torch.gather(
        eigvecs, 2, sorted_idx.unsqueeze(1).expand(-1, eigvecs.size(1), -1)
    )

    # Project features onto the top `n_components` principal components
    principal_components = eigvecs[..., :n_components]
    # b, d, k
    projected_features = torch.bmm(centered_features, principal_components)

    return projected_features


def multitensor_pca(features_list: list[torch.Tensor], 
              masks_list: list[torch.Tensor] = None, 
              n_components: int = 3):
    """
    Perform PCA on multiple feature tensors simultaneously while keeping results separable.
    
    Args:
        features_list: List of feature tensors, each shaped (b, n, d)
        masks_list: Optional list of mask tensors, each shaped (b, n)
        n_components: Number of principal components to keep
        
    Returns:
        List of projected features, each shaped (b, n, n_components)
    """
    if masks_list is None:
        masks_list = [None] * len(features_list)
    
    # Stack all features and masks for batch processing
    all_features = torch.cat(features_list, dim=1)  # (b, n_total, d)
    all_masks = torch.cat([
        mask if mask is not None else torch.ones_like(features[..., 0])
        for features, mask in zip(features_list, masks_list)
    ], dim=1)  # (b, n_total)
    
    # Perform batch PCA
    projected = torch_pca(all_features, all_masks, n_components)
    
    # Split results back to original tensors
    split_sizes = [f.shape[1] for f in features_list]
    return torch.split(projected, split_sizes, dim=1)


def torch_minmax_scale(tensor: torch.Tensor, feature_range=(0, 1)):
    """
    Scales the input tensor to the given feature range [min, max].

    Args:
        tensor (torch.Tensor): Input tensor.
        feature_range (tuple): Desired range of transformed data (default is (0, 1)).

    Returns:
        torch.Tensor: Scaled tensor.
    """
    old_shape = tensor.shape
    assert len(old_shape) >= 3, "Input tensor must have at least 3 dimensions"
    if len(old_shape) > 3:
        tensor = tensor.flatten(1, len(old_shape) - 2)
    min_val, max_val = feature_range
    tensor_min = torch.min(tensor, dim=1, keepdim=True)[0]
    tensor_max = torch.max(tensor, dim=1, keepdim=True)[0]

    # Scale the tensor to [0, 1]
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)

    # Scale to [min_val, max_val]
    scaled_tensor = scaled_tensor * (max_val - min_val) + min_val

    scaled_tensor = scaled_tensor.reshape(old_shape)
    return scaled_tensor


def dinov2_pca(features: torch.Tensor, mask=None, fg_min_norm: float = 0.5):
    bs = features.shape[0]
    features = features.permute(0, 2, 3, 1)
    feat_h, feat_w, feat_c = features.shape[1:]
    features = features.reshape(bs, feat_h * feat_w, feat_c)
    if mask is not None:
        input_mask = mask.bool().flatten(1)
        pca_features_bg = ~input_mask
    else:
        projected_features = torch_pca(features, n_components=1)
        norm_features = torch_minmax_scale(projected_features)
        # segment using the first component
        pca_features_bg = norm_features[..., 0] < fg_min_norm

    pca_features_fg = ~pca_features_bg.unsqueeze(-1)

    pca_features_rgb = torch_pca(features, n_components=3, mask=pca_features_fg)
    pca_features_rgb = torch_minmax_scale(pca_features_rgb)
    pca_features_rgb = pca_features_rgb * pca_features_fg
    pca_features_rgb = pca_features_rgb.reshape(bs, feat_h, feat_w, 3)

    return pca_features_rgb.permute(0, 3, 1, 2)
