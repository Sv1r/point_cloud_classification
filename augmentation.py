import torch
import random
import numpy as np
from torchvision import transforms


class Normalize(object):
    """Translate to origin and normalize between 0 and 1"""
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))
        return  norm_pointcloud


class RandRotation_z(object):
    """Random ratation around OZ axis"""
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        theta = random.random() * 2. * np.pi
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta),  np.cos(theta), 0],
                               [0, 0, 1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud


class RandomNoise(object):
    """Add some Gaussian noise to Point Cloud"""
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        noise = np.random.normal(0, .02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud


class ToTensor(object):
    """Convert to torch Tensor"""
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        return torch.from_numpy(pointcloud)
    

normalize = Normalize()
to_tensor = ToTensor()
rotate = RandRotation_z()
add_noise = RandomNoise()

# Train transform
train_transform = transforms.Compose(
    [normalize, rotate, add_noise, to_tensor]
    )
# Validation transform
val_transform = transforms.Compose(
    [normalize, to_tensor]
    )