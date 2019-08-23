# -*- coding: utf-8 -*-
import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tqdm import tqdm

from models.model_DanQ_JASPAR import DanQ_JASPAR
from trainer import Trainer
from loader import get_train_data, get_valid_data, get_test_data
from utils.plots import plot_loss_curve, plot_roc_curve, plot_pr_curve
from utils.metrics import calculate_auroc, calculate_aupr
from utils.utils import create_dirs, write2txt, write2csv

np.random.seed(0)
tf.random.set_seed(0)

def train():
    dataset_train = get_train_data(100)
    dataset_valid = get_valid_data(100)

    model = DanQ_JASPAR()
    loss_object = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.RMSprop()
    trainer = Trainer(
        model=model,
        loss_object=loss_object,
        optimizer=optimizer,
        experiment_dir='./result/DanQ_JASPAR')

    history = trainer.train(dataset_train, dataset_valid, epoch=32, train_steps=int(np.ceil(4400000 / 100)),
                            valid_steps=int(np.ceil(8000 / 100)), dis_show_bar=True)

    # Plot the loss curve of training and validation, and save the loss value of training and validation.
    print('\n history dict: ', history)
    epoch = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['valid_loss']
    plot_loss_curve(epoch, train_loss, val_loss, './result/DanQ_JASPAR/model_loss.jpg')
    np.savez('./result/DanQ_JASPAR/model_loss.npz', train_loss=train_loss, val_loss=val_loss)


def test():
    dataset_test = get_test_data(64)

    model = DanQ_JASPAR()
    loss_object = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.RMSprop()
    trainer = Trainer(
        model=model,
        loss_object=loss_object,
        optimizer=optimizer,
        experiment_dir='./result/DanQ_JASPAR')

    result, label = trainer.test(dataset_test, test_steps=int(np.ceil(455024 / 64)), dis_show_bar=True)

    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in tqdm(range(result_shape[1]), ascii=True):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], label[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], label[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    plot_roc_curve(fpr_list, tpr_list, './result/DanQ_JASPAR/')
    plot_pr_curve(precision_list, recall_list, './result/DanQ_JASPAR/')

    header = np.array([['auroc', 'aupr']])
    content = np.stack((auroc_list, aupr_list), axis=1)
    content = np.concatenate((header, content), axis=0)
    write2csv(content, './result/DanQ_JASPAR/result.csv')
    write2txt(content, './result/DanQ_JASPAR/result.txt')
    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    print('AVG-AUROC:{:.3f}, AVG-AUPR:{:.3f}.\n'.format(avg_auroc, avg_aupr))

if __name__ == '__main__':
    # Parses the command line arguments and returns as a simple namespace.
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('-e', '--exe_mode', default='train', help='The execution mode.')
    args = parser.parse_args()

    # Selecting the execution mode (keras).
    create_dirs(['./result/DanQ_JASPAR/'])
    if args.exe_mode == 'train':
        train()
    elif args.exe_mode == 'test':
        test()