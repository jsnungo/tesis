import os
import shutil

import argparse
import models
from sampling.sampling_utils import get_vae_sample

import soundfile as sf
import torch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of GEN AI')
parser.add_argument("--number-samples", type=int, default=10, help='number of samples')
parser.add_argument("--folder-generated-data", type=str, default='../data/', help='path of target folder')
args = parser.parse_args()
model_name = args.model
n_samples = args.number_samples

device = "cuda" if torch.cuda.is_available() else "cpu"
generation_folder = f'{args.folder_generated_data}/generated/{model_name}'

if os.path.exists(generation_folder):
    shutil.rmtree(generation_folder)
    os.makedirs(generation_folder)
else:
    os.makedirs(generation_folder)

if model_name == 'VAE':
    model = torch.load(f'{model_name}_model.pth')
    samples = get_vae_sample(model, n_samples, device)
elif model_name == 'DIFFUSSION':
    pass

sr = 16000 # TODO: ponerlo de forma m'as general
for n, s in enumerate(samples):
    to_save_audio = sf.write(f'{generation_folder}/gen_{n}.wav', s, sr)