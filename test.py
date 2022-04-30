import tensorflow as tf
import numpy as np
from network import FeatureNet
from utils.dataloader import dataloader_h5 as dataloader
import time


def test_step(x, y):
    test_logits = model(x, training=False)
    loss_value = loss_fn(y, test_logits)

    y_true = np.argmax(y.numpy(), axis=1)
    y_pred = np.argmax(test_logits.numpy(), axis=1)

    test_loss_metric.update_state(loss_value)
    test_acc_metric.update_state(y, test_logits)
    test_precision_metric.update_state(y, test_logits)
    test_recall_metric.update_state(y, test_logits)

    return y_true, y_pred


def test_step_no_labels(x):
    test_logits = model(x, training=False)
    y_pred = np.argmax(test_logits.numpy(), axis=1)

    return y_pred


if __name__ == '__main__':
    # User Parameters
    num_classes = 24
    test_set_path = "dataset/test.h5"
    checkpoint_path = "checkpoint/featurenet_date_2022-04-01.ckpt.index"

    model = FeatureNet(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    test_loss_metric = tf.keras.metrics.Mean()
    test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    test_precision_metric = tf.keras.metrics.Precision()
    test_recall_metric = tf.keras.metrics.Recall()

    model.load_weights(checkpoint_path)
    test_dataloader = dataloader(test_set_path)

    y_true_total = []
    y_pred_total = []

    start_time = time.time()

    for x_batch_test, y_batch_test in test_dataloader:
        one_hot_y = tf.one_hot(y_batch_test, depth=num_classes)
        y_true, y_pred = test_step(x_batch_test, one_hot_y)
        y_true_total = np.append(y_true_total, y_true)
        y_pred_total = np.append(y_pred_total, y_pred)

    test_loss = test_loss_metric.result()
    test_acc = test_acc_metric.result()
    test_precision = test_precision_metric.result()
    test_recall = test_recall_metric.result()

    test_loss_metric.reset_states()
    test_acc_metric.reset_states()
    test_precision_metric.reset_states()
    test_recall_metric.reset_states()

    print(f"Test loss={test_loss}, Test acc={test_acc}, Precision={test_precision}, Recall={test_recall}")
    print("Time taken: %.2fs" % (time.time() - start_time))
