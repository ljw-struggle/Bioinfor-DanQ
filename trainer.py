# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Trainer(object):
    def __init__(self, model, loss_object, optimizer, experiment_dir, patience=10, max_to_keep=5):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer

        self.patience = patience
        self.max_to_keep = max_to_keep
        self.experiment_dir = experiment_dir
        self.summary_dir = os.path.join(self.experiment_dir, 'log/')
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoint/')

        # Initialize the Metrics.
        self.metric_tra_loss = tf.keras.metrics.Mean()
        self.metric_val_loss = tf.keras.metrics.Mean()

        # Initialize the SummaryWriter.
        self.train_writer = tf.summary.create_file_writer(
            logdir=self.summary_dir + 'train/')
        self.valid_writer = tf.summary.create_file_writer(
            logdir=self.summary_dir + 'val/')

        # Initialize the CheckpointManager
        self.ckpt = tf.train.Checkpoint(
            epoch=tf.Variable(0, dtype=tf.int64),
            net=self.model,
            optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.ckpt,
            directory=self.checkpoint_dir,
            max_to_keep=self.max_to_keep)

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs=x, training=True)
            loss = self.loss_object(y_true=y, y_pred=predictions)
            loss = loss + tf.reduce_sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self, dataset_train, dataset_valid, epoch, train_steps, valid_steps, dis_show_bar=True):
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        print('Begin to train the model.\n')

        dataset_train = iter(dataset_train) # Transform the infinite iterable object to a iterator.
        dataset_valid = dataset_valid

        best_valid_loss = np.inf
        patience_temp = 0
        history = {'epoch': [], 'train_loss': [], 'valid_loss': []}

        for epoch in range(1, epoch+1):
            start_time = time.time()
            with tqdm(range(train_steps), ascii=True, disable=dis_show_bar) as pbar:
                for _, (batch_x, batch_y) in zip(pbar, dataset_train):
                    train_loss = self.train_step(batch_x, batch_y)
                    batch_size = tf.shape(batch_x)[0]
                    self.metric_tra_loss.update_state(train_loss, batch_size)
                    pbar.set_description('Train loss: {:.4f}'.format(train_loss))

            with tqdm(range(valid_steps), ascii=True, disable=dis_show_bar) as pbar:
                for _, (batch_x, batch_y) in zip(pbar, dataset_valid):
                    predictions = self.model(inputs=batch_x, training=False)
                    valid_loss = self.loss_object(y_true=batch_y, y_pred=predictions)
                    batch_size = tf.shape(batch_x)[0]
                    self.metric_val_loss.update_state(valid_loss, batch_size)
                    pbar.set_description('Valid loss: {:.4f}'.format(valid_loss))
            end_time = time.time()

            epoch_time = end_time - start_time
            real_epoch = self.ckpt.epoch.assign_add(1)
            epoch_train_loss = self.metric_tra_loss.result()
            epoch_valid_loss = self.metric_val_loss.result()
            history['epoch'].append(real_epoch.numpy())
            history['train_loss'].append(epoch_train_loss.numpy())
            history['valid_loss'].append(epoch_valid_loss.numpy())
            print("Epoch: {} | Train Loss: {:.5f}".format(real_epoch.numpy(), epoch_train_loss.numpy()), flush=True)
            print("Epoch: {} | Valid Loss: {:.5f}".format(real_epoch.numpy(), epoch_valid_loss.numpy()), flush=True)
            print("Epoch: {} | Cost time: {:.5f}: second".format(real_epoch.numpy(), epoch_time), flush=True)
            self.metric_tra_loss.reset_states()
            self.metric_val_loss.reset_states()

            # Write the summary.
            with self.train_writer.as_default():
                tf.summary.scalar('loss', epoch_train_loss, step=real_epoch)
            with self.valid_writer.as_default():
                tf.summary.scalar('loss', epoch_valid_loss, step=real_epoch)

            # Save the checkpoint. (Only save the best performance checkpoints)
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                patience_temp = 0
                save_path = self.manager.save(checkpoint_number=real_epoch)
                print("Saved checkpoint for epoch {}: {}".format(real_epoch.numpy(), save_path), flush=True)
            elif patience_temp == self.patience:
                print('Validation dice has not improved in {} epochs. Stopped training.'
                      .format(self.patience), flush=True)
                return None
            else:
                patience_temp += 1

        return history

    def test(self, dataset_test, test_steps, dis_show_bar=True):
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        results = []
        labels = []
        with tqdm(range(test_steps), ascii=True, disable=dis_show_bar, desc='Testing... ') as pbar:
            for i, (batch_x, batch_y) in zip(pbar, dataset_test):
                predictions = self.model(batch_x, training=False)
                results.append(predictions)
                labels.append(batch_y)

        results = np.concatenate(results)
        labels = np.concatenate(labels)

        return results, labels
