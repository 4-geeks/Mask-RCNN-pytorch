import os
import torch
import argparse

import utils.utils
from utils.engine import train_one_epoch, evaluate
from utils.dataset import maskrcnn_Dataset, get_transform
from utils.model import get_instance_segmentation_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='my_dataset', help='dataset path')
    parser.add_argument('--num_classes', type=int, default=11, help='number of classes (background as a class)')
    parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs')
    parser.add_argument('--batchsize', type=int, default=4, help='batchsize')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = parser.parse_args()
    
    DATASET_PATH = args.data
    num_classes = args.num_classes
    num_epochs = args.num_epochs
    batchsize = args.batchsize
    workers = args.workers
    
    
    #DATASET
    # use our dataset and defined transformations
    dataset = maskrcnn_Dataset(DATASET_PATH, get_transform(train=True))
    dataset_test = maskrcnn_Dataset(DATASET_PATH, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-int(0.3*len(dataset))])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-int(0.3*len(dataset)):])

    print('number of train data :', len(dataset))
    print('number of test data :', len(dataset_test))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batchsize, shuffle=True, num_workers=workers,
        collate_fn=utils.utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=workers,
        collate_fn=utils.utils.collate_fn)


    # MASK-RCNN MODEL
    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes).to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=15,
                                                gamma=0.1)

    # TRAINING LOOP
    
    save_fr = 1
    print_freq = 25  # make sure that print_freq is smaller than len(dataset) & len(dataset_test)
    os.makedirs('./maskrcnn_saved_models', exist_ok=True)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq)
        if epoch%save_fr == 0:
            torch.save(model.state_dict(), './maskrcnn_saved_models/mask_rcnn_model_epoch_{}.pt'.format(str(epoch)))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)