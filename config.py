import torch

# Device configuration
device_ids = [0]

# Super parameters
clamp = 2.0
log10_lr = -4.5
lr = 10 ** log10_lr
epochs = 400
weight_decay = 1e-5
device = torch.device(f"cuda:{str(device_ids[0])}" if torch.cuda.is_available() else "cpu")

# LOSS FUN
l_alpha = 3
l_beta = 1
l_gamma = 1
l_ks = 2
mse_w = 1

# Train:
batch_size = 4
cropsize = 128
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 256
batchsize_val = 1
shuffle_val = False

data_root = "./"
# Dataset
TRAIN_PATH = data_root + "data/train"
VAL_PATH = data_root + "data/test"
format_train = 'jpg'
format_val = 'jpg'

# Saving checkpoints:
MODEL_PATH = data_root + 'model/'
checkpoint_on_error = True
SAVE_freq = 50

# Load:
suffix = 'model_checkpoint_00200.pt'
tain_next = False
trained_epoch = 0

# ============================================================
#  【自动检测数据集】扫描 ./data/ 下放了哪个数据集，自动识别
#  支持: MRI-CT, MRI-PET, MRI-SPECT
#  只需把数据集文件夹放到 ./data/ 里，不用手动改任何代码
# ============================================================
import os as _os

_KNOWN_DATASETS = ['MRI-CT', 'MRI-PET', 'MRI-SPECT', 'ir-vi']

def _auto_detect_dataset():
    """返回 (规范名, 实际文件夹名)，两者相同"""
    data_dir = 'data'
    if not _os.path.exists(data_dir):
        return None, None
    existing = _os.listdir(data_dir)
    for ds in _KNOWN_DATASETS:
        if ds in existing and _os.path.isdir(_os.path.join(data_dir, ds)):
            return ds, ds
    return None, None

DATASET, DATASET_DIR = _auto_detect_dataset()

if DATASET is None:
    raise FileNotFoundError(
        f"未在 ./data/ 下找到已知数据集。\n"
        f"已知模式: {_KNOWN_DATASETS}\n"
        f"请将数据集文件夹放入 ./data/ (如 ./data/MRI-PET/)"
    )
