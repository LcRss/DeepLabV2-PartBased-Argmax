from scipy.io import loadmat
from myMetrics import *
from Utils import *
from tqdm import tqdm
from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix

"""
    Calculate mean Intersection Over Union on Validation Set every 2 epoch 
    and modify the learning rate every batch using poly learning rate formula 
"""


class CallbackmIoU(Callback):

    def __init__(self, pathTest, lr, pathRGB, pathSEG, pathArg, pathGraphs, lr_base_on_epochs=True, max_iter=0,
                 poly_lr=1):
        super().__init__()

        self.path = pathTest
        self.dir_img = pathRGB
        self.dir_seg = pathSEG
        self.dir_argmax = pathArg

        self.init_lr = lr
        self.lr_base_on_epochs = lr_base_on_epochs
        self.max_iter = max_iter
        self.poly_lr = poly_lr

        # parts list names
        self.listParts = listPartsNames()
        # dict image / class to same at the end
        self.img_dict = dictImages()
        self.classes = 108

    """
        Poly learning rate formula 
    """

    def on_train_batch_begin(self, batch, logs=None):

        if self.poly_lr:

            current_iter = tf.keras.backend.eval(self.model.optimizer.iterations)
            if current_iter != 0:

                step = self.params['steps']
                epoch_sz = self.params['epochs']

                if self.lr_base_on_epochs:
                    max_iter = epoch_sz * step
                else:
                    max_iter = self.max_iter

                up = (1 - (current_iter / max_iter)) ** 0.9
                new = self.init_lr * up

                tf.keras.backend.set_value(self.model.optimizer.lr, new)

                if current_iter % step == 0:
                    print('\nIteration %05d: reducing learning '
                          'rate to %s.' % (current_iter, new))

            else:
                tf.keras.backend.set_value(self.model.optimizer.lr, self.init_lr)
                print('\nIteration %05d: reducing learning '
                      'rate to %s.' % (current_iter, self.init_lr))

    """
        Calculate mIoU every 2 epochs 
    """

    def on_epoch_begin(self, epoch, logs=None):

        # if True:
        if epoch % 2 == 0 and epoch != 0:

            print("Begin-Callback")

            images = glob.glob(self.dir_img + "*.png")
            images.sort()

            argmax = glob.glob(self.dir_argmax + "*.png")
            argmax.sort()

            segs = glob.glob(self.dir_seg + "*.png")
            segs.sort()

            mat = np.zeros(shape=(self.classes, self.classes), dtype=np.int32)

            for k in tqdm(range(len(images))):
                img = cv2.imread(images[k])
                arg = cv2.imread(argmax[k], cv2.IMREAD_GRAYSCALE)
                seg = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)

                scale = img / 127.5 - 1

                arg = arg / 10 - 1

                arg = np.expand_dims(arg, -1)

                res = self.model.predict([np.expand_dims(scale, 0), np.expand_dims(arg, 0)])

                labels = np.argmax(res.squeeze(), -1)
                labels = labels.astype(np.uint8)

                # if (images[k])[-15:] in self.img_dict:
                #
                #     h_z, w_z = labels.shape
                #     imgNew = np.zeros((h_z, w_z, 3), np.uint8)
                #     for i in range(0, 21):
                #         mask = cv2.inRange(labels, i, i)
                #         v = self.mapLabel[i]
                #         imgNew[mask > 0] = v
                #
                #     image = make_image(imgNew)
                #     summary = tf.Summary(
                #         value=[tf.Summary.Value(tag=self.img_dict.get((images[k])[-15:]), image=image)])
                #     writer = tf.summary.FileWriter(self.path + 'logs')
                #     writer.add_summary(summary, epoch)
                #     writer.close()

                tmp = confusion_matrix(seg.flatten(), labels.flatten(), range(self.classes))

                mat = mat + tmp

            iou_108, IoU_per_class_108 = compute_and_print_IoU_per_class(confusion_matrix=mat, num_classes=108,
                                                                         namePart=self.listParts)

            print('The mIoU for epoch {} is {:7.2f}.'.format(epoch, iou_108))

    def on_train_end(self, logs=None):

        print("Begin-Callback")

        pathCMap = 'Y:/tesisti/rossi/cmap255.mat'
        fileMat = loadmat(pathCMap)
        cmap = fileMat['cmap']

        images = glob.glob(self.dir_img + "*.png")
        images.sort()

        argmax = glob.glob(self.dir_argmax + "*.png")
        argmax.sort()

        segs = glob.glob(self.dir_seg + "*.png")
        segs.sort()

        mat = np.zeros(shape=(self.classes, self.classes), dtype=np.int32)

        for k in tqdm(range(len(images))):
            img = cv2.imread(images[k])
            arg = cv2.imread(argmax[k], cv2.IMREAD_GRAYSCALE)
            seg = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)

            scale = img / 127.5 - 1

            arg = arg / 10 - 1

            arg = np.expand_dims(arg, -1)

            res = self.model.predict([np.expand_dims(scale, 0), np.expand_dims(arg, 0)])

            labels = np.argmax(res.squeeze(), -1)
            labels = labels.astype(np.uint8)

            if (images[k])[-15:] in self.img_dict:

                h_z, w_z = labels.shape
                imgNew = np.zeros((h_z, w_z, 3), np.uint8)
                for i in range(0, self.classes):
                    mask = cv2.inRange(labels, i, i)
                    v = cmap[i]
                    imgNew[mask > 0] = v

                pathTmp = self.path + (images[k])[-15:]
                cv2.imwrite(pathTmp, imgNew)

                colorSeg = np.zeros((h_z, w_z, 3), np.uint8)
                for i in range(0, self.classes):
                    mask = cv2.inRange(seg, i, i)
                    v = cmap[i]
                    colorSeg[mask > 0] = v

                pathTmp = self.path + "GT_" + (images[k])[-15:]
                cv2.imwrite(pathTmp, colorSeg)

            tmp = confusion_matrix(seg.flatten(), labels.flatten(), range(self.classes))
            mat = mat + tmp

        iou_108, IoU_per_class_108 = compute_and_print_IoU_per_class(confusion_matrix=mat, num_classes=108,
                                                                     namePart=self.listParts)

        print('The mIoU on train end is {:7.2f}.'.format(iou_108))

        create_excel_file(results21=None, results108=IoU_per_class_108, path=self.path)

        print("End at" + datetime.now().strftime("%Y%m%d-%H%M%S"))
