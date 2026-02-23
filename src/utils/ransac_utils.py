import torch
import numpy as np
import random


def np2pcd(points):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def np2pcf(feat):
    import open3d as o3d

    pcf = o3d.pipelines.registration.Feature()
    pcf.data = feat.T
    return pcf


def tensor2pcd(points):
    return np2pcd(points.cpu().numpy())


def tensor2pcf(feat):
    return np2pcf(feat.cpu().numpy())
