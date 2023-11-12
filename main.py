import os
import torch
import random
import numpy as np

import utils
import settings
import dataset
import model_build


def main():
    # Fix random behavior
    random.seed(settings.RANDOM_STATE)
    np.random.seed(settings.RANDOM_STATE)
    torch.manual_seed(settings.RANDOM_STATE)
    torch.cuda.manual_seed(settings.RANDOM_STATE)

    # Model
    model = model_build.PointNet()
    # Initialize model weights
    model.apply(utils.init_weights)
    # Optimizer with weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE)
    # Scheduler
    scheduler = None
    # Save results folders
    path_to_save = './runs'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    path_to_train = f'{path_to_save}/exp'
    if not os.path.exists(path_to_train):
        folder_to_save_weights = f'{path_to_train}/weights'
        for direction in [path_to_train, folder_to_save_weights]:
            os.makedirs(direction)
    else:
        count_folders = 1
        while os.path.exists(path_to_train + f'{count_folders}'):
            count_folders += 1
        path_to_train += f'{count_folders}'
        folder_to_save_weights = f'{path_to_train}/weights'
        for direction in [path_to_train, folder_to_save_weights]:
            os.makedirs(direction)
    # Train function
    model, df = utils.train_model(
        folder_to_save_weights=folder_to_save_weights,
        path_to_train=path_to_train,
        model=model,
        train_dataloader=dataset.train_dataloader,
        valid_dataloader=dataset.valid_dataloader,
        loss=utils.pointnetloss,
        optimizer=optimizer,
        num_epoch=settings.EPOCH,
        scheduler=scheduler,
        avg_precision=True
    )
    # Plot results
    utils.result_plot(data=df, path_to_train=path_to_train)


if __name__ == '__main__':
    main()