import sys
import cv2
import numpy as np
import tensorflow as tf


class adj_mat_func(object):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def adj_mat(self, y_true, y_pred):
        # Wraps np_mean_iou method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self.np_adj_func, [y_true, y_pred], tf.float32)

    def np_adj_func(self, y_true, y_pred):

        adj_mat = np.zeros(shape=(108, 108))
        for o in range(self.batch_size):
            img = y_true[o]
            classes = np.unique(img)
            classes = classes[1:]
            if 255 in classes:
                classes = classes[:-1]
            mat_contour = []

            for i in range(len(classes)):

                value = classes[i]
                mask = cv2.inRange(img, int(value), int(value))
                _, per, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

                mat_total = np.zeros(shape=(1, 2))

                for q in range(len(per)):

                    tmp = per[q]
                    mat = np.zeros(shape=(len(tmp), 2))
                    for j in range(len(tmp)):
                        point = tmp[j]
                        x = point[0][0]
                        y = point[0][1]
                        mat[j][0] = x
                        mat[j][1] = y

                    mat_total = np.concatenate((mat_total, mat), axis=0)

                mat_contour.append(mat_total[1:])

            for i in range(len(classes)):
                tmp = mat_contour[i]

                for j in range(i + 1, len(classes)):

                    min_v = sys.maxsize
                    second_mat = mat_contour[j]

                    for p in range(len(tmp)):
                        first_mat = tmp[p]

                        dif = first_mat - second_mat
                        dif = dif * dif
                        sum_mat = np.sum(dif, 1)
                        sqrt = np.sqrt(sum_mat)

                        min_tmp = np.min(sqrt)
                        if min_tmp < min_v:
                            min_v = min_tmp

                    if min_v <= 1:
                        adj_mat[classes[i]][classes[j]] = 1 + adj_mat[classes[i]][classes[j]]

        return adj_mat.astype(np.float32)

    def np_adj_func_2in1(self, y_true, y_pred):

        adj_mat_true = np.zeros(shape=(108, 108))
        for o in range(self.batch_size):
            img = y_true[o]
            classes = np.unique(img)
            classes = classes[1:]
            if 255 in classes:
                classes = classes[:-1]
            mat_contour = []

            for i in range(len(classes)):

                value = classes[i]
                mask = cv2.inRange(img, int(value), int(value))
                _, per, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

                mat_total = np.zeros(shape=(1, 2))

                for q in range(len(per)):

                    tmp = per[q]
                    mat = np.zeros(shape=(len(tmp), 2))
                    for j in range(len(tmp)):
                        point = tmp[j]
                        x = point[0][0]
                        y = point[0][1]
                        mat[j][0] = x
                        mat[j][1] = y

                    mat_total = np.concatenate((mat_total, mat), axis=0)

                mat_contour.append(mat_total[1:])

            for i in range(len(classes)):
                tmp = mat_contour[i]

                for j in range(i + 1, len(classes)):

                    min_v = sys.maxsize
                    second_mat = mat_contour[j]

                    for p in range(len(tmp)):
                        first_mat = tmp[p]

                        dif = first_mat - second_mat
                        dif = dif * dif
                        sum_mat = np.sum(dif, 1)
                        sqrt = np.sqrt(sum_mat)

                        min_tmp = np.min(sqrt)
                        if min_tmp < min_v:
                            min_v = min_tmp

                    if min_v <= 1:
                        adj_mat_true = [classes[i]][classes[j]] = 1 + adj_mat_true[classes[i]][classes[j]]

        adj_mat_pred = np.zeros(shape=(108, 108))
        for o in range(self.batch_size):
            img = y_pred[o]
            classes = np.unique(img)
            classes = classes[1:]
            if 255 in classes:
                classes = classes[:-1]
            mat_contour = []

            for i in range(len(classes)):

                value = classes[i]
                mask = cv2.inRange(img, int(value), int(value))
                _, per, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

                mat_total = np.zeros(shape=(1, 2))

                for q in range(len(per)):

                    tmp = per[q]
                    mat = np.zeros(shape=(len(tmp), 2))
                    for j in range(len(tmp)):
                        point = tmp[j]
                        x = point[0][0]
                        y = point[0][1]
                        mat[j][0] = x
                        mat[j][1] = y

                    mat_total = np.concatenate((mat_total, mat), axis=0)

                mat_contour.append(mat_total[1:])

            for i in range(len(classes)):
                tmp = mat_contour[i]

                for j in range(i + 1, len(classes)):

                    min_v = sys.maxsize
                    second_mat = mat_contour[j]

                    for p in range(len(tmp)):
                        first_mat = tmp[p]

                        dif = first_mat - second_mat
                        dif = dif * dif
                        sum_mat = np.sum(dif, 1)
                        sqrt = np.sqrt(sum_mat)

                        min_tmp = np.min(sqrt)
                        if min_tmp < min_v:
                            min_v = min_tmp

                    if min_v <= 1:
                        adj_mat_pred = [classes[i]][classes[j]] = 1 + adj_mat_pred[classes[i]][classes[j]]

        return adj_mat_true.astype(np.float32), adj_mat_pred.astype(np.float32)
