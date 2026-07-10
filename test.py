import torch
import torchvision
from model import Model
import config as c
import os
import shutil
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import time
import torch.nn.functional as F
from tqdm import tqdm

cuda_my = f"cuda:{str(c.device_ids[0])}"
device = torch.device(cuda_my if torch.cuda.is_available() else "cpu")

def load(net, name):
    try:
        state_dicts = torch.load(name, map_location=cuda_my, weights_only=True)
    except TypeError:
        state_dicts = torch.load(name, map_location=cuda_my)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    network_state_dict = {k.replace('.darm.', '.cmsa.'): v for k, v in network_state_dict.items()}
    network_state_dict = {k.replace('.cmi.', '.mac.'): v for k, v in network_state_dict.items()}
    net.load_state_dict(network_state_dict)

def pad_to_multiple(x, multiple=8):
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    return x, (pad_h, pad_w)

def test(dataset_name, data_root, test_out_path, data_folder_name=None):
    if data_folder_name is None:
        data_folder_name = dataset_name
    Time = []
    test_folder = os.path.join(data_root, data_folder_name)  # 实际数据文件夹
    test_out_folder = os.path.join(test_out_path, dataset_name)  # 输出用规范名
    # 每次运行前清空并重建输出文件夹
    if os.path.exists(test_out_folder):
        shutil.rmtree(test_out_folder)
    os.makedirs(test_out_folder)
    
    ds_key = dataset_name.rstrip('-')  # 兼容 MRI-CT- 等命名
    if ds_key in ['ir-vi', 'IR-VIS']:
        model_path = './model/model-1-best.pt'
    elif ds_key in ['MRI-CT', 'MRI-PET', 'MRI-SPECT']:
        model_path = './model/model-1-best.pt'
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    net = Model()
    net.to(device)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    load(net, model_path)
    net.eval()

    subfolders = sorted([
        folder_name for folder_name in os.listdir(test_folder)
        if os.path.isdir(os.path.join(test_folder, folder_name))
    ])
    if len(subfolders) != 2:
        raise ValueError(f"Expected exactly 2 subfolders in {test_folder}, got {subfolders}")

    if 'MRI' in subfolders:
        mri_folder = 'MRI'
        other_folder = subfolders[0] if subfolders[1] == 'MRI' else subfolders[1]
    else:
        mri_folder, other_folder = subfolders[0], subfolders[1]

    img_names = sorted(os.listdir(os.path.join(test_folder, mri_folder)))
    for img_name in tqdm(img_names, desc=f"测试 {dataset_name}", unit="张"):
        other_path = os.path.join(test_folder, other_folder, img_name)
        MRI_path = os.path.join(test_folder, mri_folder, img_name)
        OTHER_img = Image.open(other_path).convert("RGB")
        MRI_img = Image.open(MRI_path).convert("RGB")

        crop_h = min(MRI_img.height, OTHER_img.height)
        crop_w = min(MRI_img.width, OTHER_img.width)
        center_crop = T.CenterCrop((crop_h, crop_w))
        MRI_img = center_crop(MRI_img)
        OTHER_img = center_crop(OTHER_img)

        data0 = ToTensor()(MRI_img).unsqueeze(0)
        data1 = ToTensor()(OTHER_img).unsqueeze(0)

        tic = time.time()
        with torch.no_grad():
            cover = data0.to(device)
            secret = data1.to(device)
            orig_h, orig_w = cover.shape[-2], cover.shape[-1]
            cover, _ = pad_to_multiple(cover, multiple=8)
            secret, _ = pad_to_multiple(secret, multiple=8)
            input_img = torch.cat((cover, secret), 1)
            output = net(input_img)
            output = output[..., :orig_h, :orig_w]
        end = time.time()
        Time.append(end - tic)
        torchvision.utils.save_image(output, os.path.join(test_out_folder, img_name))

        del output, secret, cover, MRI_img, OTHER_img
        torch.cuda.empty_cache()

    Time = Time[2:len(Time) - 2]
    return (sum(Time))

if __name__ == '__main__':
    data_root = f'./data'
    test_out_folder = f'./result'
    dataset_name = c.DATASET       # 规范名，如 'MRI-CT'，用于输出命名
    data_folder_name = c.DATASET_DIR  # 实际文件夹名，如 'MRI-CT-'

    print(f"当前数据集: {dataset_name} (文件夹: {data_folder_name})")
    test_time_avg = test(dataset_name, data_root, test_out_folder, data_folder_name)
    print(test_time_avg)
