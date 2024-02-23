import argparse
import json
import models
import training_loss

import torch
from dataset_manager import GenerativeAIDataset
from torch.utils.data import DataLoader

from transformation.data_preprocessing import get_data_composing

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train-dataset", type=str, default='../data/train', help='path of train dataset')
parser.add_argument("--valid-dataset", type=str, default='../data/val', help='path of validation dataset')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--learning-rate", type=float, default=1e-3, help='learning rate for optimization')
parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
# parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.') # para el otro tipo de 
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=70, help='max number of epochs')
parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of GEN AI')
parser.add_argument("--config", type=str, default='./config.json', help='path of difussion config')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)


device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch_size
model_name = args.model
lr = args.learning_rate

data_processing = get_data_composing(model_name, config)
train_dataset = GenerativeAIDataset(args.train_dataset,
                            data_processing,
                            class_c='BOAFAB')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)

model = models.create_model(model_name, config=config).to(device)
training_loss_function = training_loss.get_loss_train(model_name, general_config=config)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if args.lr_scheduler == 'plateau':
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)


def train(epoch):
    
    model.train()  # Set model to training mode (lo dejo si hago validación)

    running_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        batch = batch['samples'].to(device)
        optimizer.zero_grad()
        loss = training_loss_function(model, batch)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

        running_loss += loss.item()

    loss_epoch = running_loss / len(train_dataloader)

    return loss_epoch

print(f'Training {model_name}')

for epoch in range(args.max_epochs):

    loss_train = train(epoch)

    if args.lr_scheduler == 'plateau':
        lr_scheduler.step(metrics=loss_train)

torch.save(model, f'{model_name}_model.pth')
print('Model was save')