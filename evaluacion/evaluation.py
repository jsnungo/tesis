import argparse
import json
import sys
import os
sys.path.append(os.path.abspath("../classificador"))

import utils_fid as uf
import torch
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--real-dataset", type=str, default='../data/train', help='path of train dataset')
parser.add_argument("--generated-dataset", type=str, default='../data/generated/VAE', help='path of generated dataset')
parser.add_argument("--classifier-model", type=str, default='../classificador/1707962574001-resnext29_8_64_sgd_plateau_bs2_lr1.0e-02_wd1.0e-02-best-loss.pth', help='model to perform the embedding')
parser.add_argument("--size-sample", type=int, default=1000, help='number of samples to get gaussioan estimation')
parser.add_argument("--get-reduction", type=bool, default=True, help='model to perform the embedding')
parser.add_argument("--dim-reduction", type=int, default=2, help='model to perform the embedding')
args = parser.parse_args()

path_model = args.classifier_model
print(path_model)
model = torch.load(path_model)

path_gen_data = args.generated_dataset
path_real_data = args.real_dataset
size_sample = args.size_sample

gen_data = uf.get_embedding(model, path_gen_data, class_c='BOAFAB', sample_size=size_sample)
real_data = uf.get_embedding(model, path_real_data, class_c='BOAFAB', sample_size=size_sample)

fid_score = uf.calculate_fid(real_data, gen_data)

print('FID SCORE ::: ', fid_score)

if args.get_reduction:
    data = np.vstack((
        real_data,
        gen_data
    ))
    grouper = ['Real'] * len(real_data) + ['Generado'] * len(gen_data)
    scaled_real_data = scale(data)

    dim_new_dimension = args.dim_reduction

    pca = PCA(n_components=dim_new_dimension)
    pca.fit(scaled_real_data)

    data_transformed = pca.transform(scaled_real_data)

    sns.scatterplot(
        x=data_transformed[:, 0],
        y=data_transformed[:, 1],
        hue=grouper
    )

    plt.show()
