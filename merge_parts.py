from DeeplabV2_resnet101_params import ResNet101
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from myMetrics import *
from Utils import *
import argparse

n_class = 108

parser = argparse.ArgumentParser()
parser.add_argument("--check_point", type=str, default='not_defined')
parser.add_argument("--path", type=str, default='not_defined')
args = parser.parse_args()

pathCP ="H:\Test_Tesi_ordinati\Argmax\Argmax_standard_loss_lambda_1_kern_init_glorot_normal_lr_0.001_batch_2_use_mult_rate_2\Checkpoints_deeplab/checkpoint-14-1.27.hdf5"
pathSave = args.path

assert not (pathCP == 'not_defined'), "Checkpoint path not defined"
if pathSave == 'not_defined':
    pathSave = "./"
else:
    pathSave = pathSave + "/"

pT = pathSave + datetime.now().strftime("%Y%m%d-%H%M%S")
print(pathCP)

if not os.path.isdir(pT):
    os.mkdir(pT)
inf_path = pT + "/inf"
os.mkdir(inf_path)

print("load model weights")
deeplab_model = ResNet101(input_shape_1=(None, None, 3), input_shape_2=(None, None, 1), classes=n_class)
deeplab_model.load_weights(pathCP, by_name=True)

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

mat = np.zeros(shape=(21, 21), dtype=np.int32)
mapParts = mapCl2Prt()

for k in tqdm(range(len(argmax))):
    # for k in tqdm(range(2)):
    img = cv2.imread(images[k])
    arg = cv2.imread(argmax[k], cv2.IMREAD_GRAYSCALE)
    seg = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)

    h_z, w_z = seg.shape
    w, h, _ = img.shape
    mask = seg != 255
    seg = seg[mask]

    scale = img / 127.5 - 1
    arg = arg / 10 - 1
    arg = np.expand_dims(arg, -1)

    res = deeplab_model.predict([np.expand_dims(scale, 0), np.expand_dims(arg, 0)])
    labels = np.argmax(res.squeeze(), -1)
    labels = labels.astype(np.uint8)
    labels_mask = labels[mask]

    for class21 in range(0, 21):
        parts = mapParts[class21]
        for p in range(parts[0], parts[1]):
            mask = (labels_mask == p)
            if np.sum(mask) > 0:
                labels_mask[mask] = class21 + 108

    for class21 in range(0, 21):
        parts = mapParts[class21]
        for p in range(parts[0], parts[1]):
            mask = (seg == p)
            if np.sum(mask) > 0:
                seg[mask] = class21 + 108

    for class21 in range(0, 21):
        parts = mapParts[class21]
        for p in range(parts[0], parts[1]):
            mask = (labels == p)
            if np.sum(mask) > 0:
                labels[mask] = class21 + 108

    labels = labels - 108
    seg = seg - 108
    labels_mask = labels_mask - 108

    tmp = confusion_matrix(seg, labels_mask, range(21))
    mat = mat + tmp

    # TO color labels and save image

    imgNew = np.zeros((h_z, w_z, 3), np.uint8)
    for i in range(0, n_class):
        mask = cv2.inRange(labels, i, i)
        v = cmap[i]
        imgNew[mask > 0] = v

    name = images[k]
    name = name[-15:]

    pathTmp = inf_path + "/" + name
    cv2.imwrite(pathTmp, imgNew)
    # cv2.imshow("a", imgNew)
    # cv2.waitKey()

#
classes = listClassesNames()
mIoU, IoU_per_class = compute_and_print_IoU_per_class(confusion_matrix=mat, num_classes=21, namePart=classes)
create_excel_file(results21=IoU_per_class, path=pT + "/")
