import numpy as np
import matplotlib.pyplot as plt

import utils
import augmentation


def plot_scatter(xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


class Normalize(object):
    """Translate to origin and normalize between 0 and 1"""
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= max(np.linalg.norm(norm_pointcloud, axis=1))
        return  norm_pointcloud
    

if __name__ == '__main__':
    
    with open('/app/3d_classification/data/ModelNet10/ModelNet10/desk/test/desk_0204.off', 'r') as f:
        verts, faces = utils.read_off(f)

    point_sampler = utils.PointSampler(3000)
    pointcloud = point_sampler((verts, faces))
    # Define augmentation methods
    normalize = augmentation.Normalize()
    rotate = augmentation.RandRotation_z()
    add_noise = augmentation.RandomNoise()
    # Apply them to the point cloud
    pointcloud = normalize(pointcloud)
    pointcloud = rotate(pointcloud)
    pointcloud = add_noise(pointcloud)
    x, y, z = pointcloud.T
    plot_scatter(xs=x, ys=y, zs=z)
