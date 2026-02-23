from pytorch3d.structures import join_meshes_as_batch


def repeat_interleave_meshes(meshes, n):
    meshes_list = []
    for m in meshes:
        meshes_list.extend([m for _ in range(n)])
    return join_meshes_as_batch(meshes_list)


def repeat_meshes(mesh, n):
    return join_meshes_as_batch([mesh for _ in range(n)])
