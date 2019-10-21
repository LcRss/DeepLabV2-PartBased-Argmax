import cv2
import glob
from scipy.io import loadmat

from DeeplabV2_resnet101_params import ResNet101
from myMetrics import *
from Utils import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

n_class = 108

print("load model weights")
deeplab_model = ResNet101(input_shape_1=(None, None, 3), input_shape_2=(None, None, 1), classes=n_class)

path = "F:/checkpoint-14-1.27.hdf5"
deeplab_model.load_weights(path, by_name=True)

# LAB
dir_img = "Y:/tesisti/rossi/data/train_val_test_png/test_png/"
dir_argmax = "Y:/tesisti/rossi/data/segmentation_gray/results_deeplabV2/test/"
dir_seg = 'Y:/tesisti/rossi/data/segmentation_part_gray/new_dataset_107\data_part_107part_test/'
####

images = glob.glob(dir_img + "*.png")
images.sort()
argmax = glob.glob(dir_argmax + "*.png")
argmax.sort()
segs = glob.glob(dir_seg + "*.png")
segs.sort()

pathCMap = 'Y:/tesisti/rossi/cmap255.mat'
fileMat = loadmat(pathCMap)
cmap = fileMat['cmap']

mat = np.zeros(shape=(n_class, n_class), dtype=np.int32)

for k in tqdm(range(len(images))):

    img = cv2.imread(images[k])
    arg = cv2.imread(argmax[k], cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("i", img)
    # cv2.waitKey()

    h_z, w_z = seg.shape
    segNew = np.zeros((h_z, w_z, 3), np.uint8)
    for i in range(0, n_class):
        mask = cv2.inRange(seg, i, i)
        v = cmap[i]
        segNew[mask > 0] = v

    # cv2.imshow("s", segNew)
    # cv2.waitKey()

    w, h, _ = img.shape
    scale = img / 127.5 - 1
    arg = arg / 10 - 1
    arg = np.expand_dims(arg, -1)
    # print(scale.shape)

    # print(newScale.shape)

    res = deeplab_model.predict([np.expand_dims(scale, 0), np.expand_dims(arg, 0)])
    labels = np.argmax(res.squeeze(), -1)
    z = labels.astype(np.uint8)

    tmp = confusion_matrix(seg.flatten(), z.flatten(), range(n_class))
    mat = mat + tmp

    h_z, w_z = z.shape
    imgNew = np.zeros((h_z, w_z, 3), np.uint8)
    for i in range(0, n_class):
        mask = cv2.inRange(z, i, i)
        v = cmap[i]
        imgNew[mask > 0] = v

    name = images[k]
    name = name[-15:]
    

    pathTmp = "D:/rossi/argmax/Argmax_standard_loss_lambda_1_kern_init_glorot_uniform_lr_0.001_batch_2_use_BN/" + name
    cv2.imwrite(pathTmp, imgNew)

#
listParts = listPartsNames()
iou = compute_and_print_IoU_per_class(confusion_matrix=mat, num_classes=n_class, namePart=listParts)
print(iou)
