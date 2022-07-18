# -*- coding: utf-8 -*-
import os
import argparse
from keras.callbacks import EarlyStopping
from keras import losses
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta
from sklearn.metrics.pairwise import paired_distances as dist
from Hyper import imgDraw, listClassification, resnet99_avg_recon
import libmr
import numpy as np
import rscls
import glob
from copy import deepcopy

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--numTrain', type=int, default=20)  # number of training samples per class,small than 40
parser.add_argument('--dataset', type=str, default='data/trento_im.npy')  # dataset name
parser.add_argument('--gt', type=str, default='data/trento_raw_gt.npy')  # only known training samples included
parser.add_argument('--batch_size', type=int, default=16)  # only known training samples included
parser.add_argument('--output', type=str, default='output/')  # save path for output files
args = parser.parse_args()

# generate output dir
early_stopping = EarlyStopping(monitor='loss', patience=1000)
key = args.dataset.split('/')[-1].split('_')[0]
spath = args.output + key + '_' + str(args.numTrain) + '/'
os.makedirs(spath, exist_ok=True)

# load dataset
print('Preparing dataset...')
hsi = np.load(args.dataset).astype('float32')
gt = np.load(args.gt).astype('int')
listClassification(gt)
numClass = gt.max()
row, col, layers = hsi.shape
hsi = np.float32(hsi)
# dataset format
c1 = rscls.rscls(hsi, gt, cls=numClass)
c1.padding(9)
x1_train, y1_train = c1.train_sample(args.numTrain)  # load train samples
x1_train, y1_train = rscls.make_sample(x1_train, y1_train)  # augmentation
x1_lidar_train = x1_train[:, :, :, -1]
x1_lidar_train = x1_lidar_train[:, :, :, np.newaxis]
x1_train = x1_train[:, :, :, :-1]
y1_train = to_categorical(y1_train, numClass)  # to one-hot labels

print('Start training...')
model, _ = resnet99_avg_recon(layers - 1, 1, 9, numClass, l=1)
model.compile(loss=['categorical_crossentropy', losses.mean_absolute_error], optimizer=Adadelta(learning_rate=1.0),
              metrics=['accuracy'], loss_weights=[0.5, 0.5])
model.fit([x1_train, x1_lidar_train], [y1_train, x1_train], batch_size=args.batch_size,
          epochs=270, verbose=1, shuffle=True, callbacks=[early_stopping])
model.compile(loss=['categorical_crossentropy', losses.mean_absolute_error], optimizer=Adadelta(learning_rate=0.1),
              metrics=['accuracy'], loss_weights=[0.5, 0.5])
model.fit([x1_train, x1_lidar_train], [y1_train, x1_train], batch_size=args.batch_size,
          epochs=230, verbose=1, shuffle=True, callbacks=[early_stopping])
model.save(spath + args.dataset + '_model')

print('Start predicting...')
pre_all = []
pre_loss = []
for r in tqdm(range(row)):
    row_samples = c1.all_sample_row(r)
    pre_row, recons = model.predict([row_samples[:, :, :, :-1], (row_samples[:, :, :, -1])[:, :, :, np.newaxis]])
    pre_all.append(pre_row)
    recons_loss = dist(recons.reshape(col, -1), row_samples[:, :, :, :-1].reshape(col, -1))
    pre_loss.append(recons_loss)
pre_all = np.array(pre_all).astype('float64')
pre_loss = np.array(pre_loss).reshape(-1).astype('float64')
recons_train = model.predict([x1_train, x1_lidar_train])[1]
train_loss = dist(recons_train.reshape(recons_train.shape[0], -1), x1_train.reshape(x1_train.shape[0], -1))

print('Start caculating open-set...')
mr = libmr.MR()
mr.fit_high(train_loss, 20)
wscore = mr.w_score_vector(pre_loss)
mask = wscore > 0.5  # default threshold=0.5
mask = mask.reshape(row, col)
unknown = gt.max() + 1

# for close set
pre_closed = np.argmax(pre_all, axis=-1) + 1  # baseline: closed
imgDraw(pre_closed, spath + key + '_closed', path='./', show=False)

# for open set
pre_gsrl = deepcopy(pre_closed)
pre_gsrl[mask == 1] = unknown  # unkown class
gt_new = deepcopy(gt)
gt2file = glob.glob('data/' + key + '*gt*[0-9].npy')[0]
gt2 = np.load(gt2file)
gt_new[np.logical_and(gt_new == 0, gt2 != 0)] = unknown
cfm = rscls.gtcfm(pre_gsrl, gt_new, unknown)

pre_to_draw = deepcopy(pre_gsrl)
pre_to_draw[pre_to_draw == unknown] = 0
imgDraw(pre_to_draw, spath + key + '_gsrl', path='./', show=False)
