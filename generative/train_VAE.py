import argparse
import time
import models

import torch
from dataset_manager import GenAIDataset
from torch.utils.data import DataLoader

from transformation.data_preprocessing import get_data_composing


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train-dataset", type=str, default='../data/train', help='path of train dataset')
parser.add_argument("--valid-dataset", type=str, default='../data/val', help='path of validation dataset')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--learning-rate", type=float, default=1e-4, help='learning rate for optimization')
parser.add_argument("--max-epochs", type=int, default=70, help='max number of epochs')
parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of GEN AI')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch_size
model_name = args.model

assert model_name in models.available_models


data_processing = get_data_composing(model_name)

train_dataset = GenAIDataset(args.train_dataset,
                            data_processing,
                            class_c='BOAFAB')
