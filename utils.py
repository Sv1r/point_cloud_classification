import glob
import torch
import random
import numpy as np
import pandas as pd


def data_list(path_to_data):
    # List of data
    list_of_data = glob.glob(f'{path_to_data}/*/*/*')
    # Create dataframe
    df = pd.DataFrame(list_of_data, columns=['path_to_cad'])
    # Add class
    df['class'] = df['path_to_cad'].apply(lambda x: x.split(sep='/')[-3])
    # Unique classes
    classes_names = df['class'].unique().tolist()
    # Class dict with ids
    classes_dict = dict(
        zip(
            classes_names, list(range(len(classes_names)))
            )
        )
    # Add class id
    df['class_id'] = df['class'].apply(lambda x: classes_dict[x])
    # Add subset marker
    df['subset'] = df['path_to_cad'].apply(lambda x: 1 if 'train' in x else 0)
    # Train/val split
    train = df.loc[df['subset'] == 1]
    val = df.loc[df['subset'] == 0]
    return train, val


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** .5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (
                self.triangle_area(
                    verts[faces[i][0]],
                    verts[faces[i][1]], 
                    verts[faces[i][2]]
                    )
                )
        sampled_faces = (
            random.choices(
                faces,
                weights=areas, 
                cum_weights=None, 
                k=self.output_size
                )
            )
        sampled_points = np.zeros((self.output_size, 3))
        for i in range(len(sampled_faces)):
            sampled_points[i] = (
                self.sample_point(
                    verts[sampled_faces[i][0]],
                    verts[sampled_faces[i][1]], 
                    verts[sampled_faces[i][2]]
                    )
                )
        return sampled_points


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, data, number_of_points, transform=None):
        self.paths_to_cad = data['path_to_cad'].tolist()
        self.categories = data['class_id'].tolist()
        self.number_of_points = number_of_points
        self.transform = transform

    def __len__(self):
        return len(self.paths_to_cad)

    def __getitem__(self, index):
        path_to_cad = self.paths_to_cad[index]
        category = self.categories[index]

        with open(path_to_cad, 'r') as f:
            verts, faces = read_off(f)
        # Define sampler
        point_sampler = PointSampler(self.number_of_points)
        # Point Cloud from verts
        pointcloud = point_sampler((verts, faces))
        # Augmentations
        if self.transform is not None:
            pointcloud = self.transform(pointcloud)
        return pointcloud, torch.tensor(category)
    
