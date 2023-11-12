import glob
import tqdm
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
    return train.head(32), val.head(32)


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
    

def init_weights(m):
    try:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(.01)
    except Exception:
        pass


def train_model(
        folder_to_save_weights,
        path_to_train,
        model,
        train_dataloader,
        valid_dataloader,
        loss,
        optimizer,
        num_epoch,
        scheduler=None,
        avg_precision=True,
):
    """Model training function"""
    train_loss_history, valid_loss_history = [], []
    # Dataframe
    df = pd.DataFrame()
    # Model to device
    model = model.to(device)
    # Scaler for average precision training
    scaler = torch.cuda.amp.GradScaler(enabled=avg_precision)
    # Initial minimum loss
    valid_min_loss = 1e10

    for epoch in range(num_epoch):
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = valid_dataloader
                model.eval()  # Set model to evaluate mode
            running_loss = []
            # Iterate over data.
            with tqdm.tqdm(dataloader, unit='batch') as tqdm_loader:
                for inputs, labels in tqdm_loader:
                    tqdm_loader.set_description(f'Epoch {epoch}/{num_epoch-1} - {phase}')
                    # Point Cloud and Label to device
                    inputs = inputs.to(device, dtype=torch.half)
                    labels = labels.to(device, dtype=torch.half)
                    optimizer.zero_grad()
                    # forward and backward
                    with torch.set_grad_enabled(phase == 'Train'):
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=avg_precision):
                            predict, m3x3, m64x64 = model(inputs.transpose(1, 2))
                            loss_value = loss(predict, labels, m3x3, m64x64)
                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            scaler.scale(loss_value).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            if scheduler is not None:
                                scheduler.step()
                    # statistics
                    loss_item = loss_value.item()
                    running_loss.append(loss_item)
                    # Current statistics
                    tqdm_loader.set_postfix(Loss=loss_item)
                    time.sleep(.1)
            epoch_loss = np.mean(running_loss)
            # Checkpoint
            if epoch_loss < valid_min_loss and phase != 'Train':
                valid_min_loss = epoch_loss
                model = model.cpu()
                torch.save(model, f'{folder_to_save_weights}/best.pt')
                model = model.to(device)
            # Loss history
            if phase == 'Train':
                train_loss_history.append(epoch_loss)
            else:
                valid_loss_history.append(epoch_loss)
            print(
                'Epoch: {}/{}  Stage: {} Loss: {:.6f}'.format(
                    epoch, num_epoch-1, phase, epoch_loss
                ), flush=True
            )
            time.sleep(.1)

    # Add results for each model
    df['Train_Loss'] = train_loss_history
    df['Valid_Loss'] = valid_loss_history
    # Save df if csv format
    df.to_csv(f'{path_to_train}/results.csv', sep=' ', index=False)
    # Save last model
    torch.save(model, f'{folder_to_save_weights}/last.pt')

    return model, df


def result_plot(data, path_to_train):
    """Plot loss function and Metrics"""
    stage_list = np.unique(list(map(lambda x: x.split(sep='_')[0], data.columns)))
    variable_list = np.unique(list(map(lambda x: x.split(sep='_')[1], data.columns)))
    plt.subplots(figsize=(10, 10))
    for stage in stage_list:
        for variable in variable_list:
            plt.plot(data[f'{stage}_{variable}'], label=f'{stage}')
            plt.title(f'{variable} Plot', fontsize=10)
            plt.xlabel('Epoch', fontsize=8)
            plt.ylabel(f'{variable} Value', fontsize=8)
            plt.legend()
    plt.savefig(f'{path_to_train}/loss_plot.png')


def pointnetloss(outputs, labels, m3x3, m64x64, alpha = .0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1, 2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)
