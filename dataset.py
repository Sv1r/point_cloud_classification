from torch.utils.data import DataLoader

import utils
import settings
import augmentation

# Prepare pandas dataframes
train, val = utils.data_list(
    path_to_data=settings.PATH_TO_DATA
    )
# Datasets
train_dataset = utils.PointCloudDataset(
    data=train, 
    number_of_points=settings.NUMBER_OF_POINTS, 
    transform=augmentation.train_transform
    )
valid_dataset = utils.PointCloudDataset(
    data=val, 
    number_of_points=settings.NUMBER_OF_POINTS, 
    transform=augmentation.val_transform
    )
# Dataloaders
train_dataloader = DataLoader(
    dataset=train_dataset, 
    batch_size=settings.BATCH_SIZE, 
    shuffle=True
    )
valid_dataloader = DataLoader(
    dataset=valid_dataset, 
    batch_size=settings.BATCH_SIZE, 
    shuffle=True
    )

if __name__ == '__main__':
    print(f'All possible classes: {train["class"].unique().shape[0]}')
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Valid dataset size: {len(valid_dataset)}')
    print(f'Sample pointcloud shape: {list(train_dataset[0][0].shape)}')
    print(f'Class: {train_dataset[0][1]}')
