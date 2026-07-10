# ASFFuse User Guide

## Quick Test

### 1. Prepare Data

Place the infrared images you want to test into:

```
data/ir-vi/ir/
```

Place the corresponding visible images into:

```
data/ir-vi/vi/
```

**Requirements:**
- The image file names in the two folders must match one-to-one.
- Images are automatically cropped to the same size; it is recommended to pre-align the image pairs.
- Common formats such as `PNG` and `JPG` are supported.

### 2. Run the Test

Execute the following command in the project root directory:

```bash
python test.py
```

The program will automatically detect the `data/ir-vi` directory and load the pre-trained model `./model/model-1-best.pt` for inference.

### 3. View the Results

The fusion results will be saved in:

```
result/ir-vi/
```

Each input image pair produces one corresponding fused image, with the same file name as the input infrared/visible images.

## Directory Structure

```
ASFFuse/
├── data/
│   └── ir-vi/
│       ├── ir/          # infrared images
│       └── vi/          # visible images
├── model/
│   └── model-1-best.pt  # pre-trained weights
├── result/
│   └── ir-vi/           # fused result output
├── test.py              # test script
└── config.py            # configuration file
```

## Notes

- The test script uses the GPU device configured in `config.py` by default. If GPU memory is insufficient, change `device_ids` in `config.py` to a suitable GPU index, or the program will automatically fall back to CPU.
- `result/ir-vi/` is cleared and recreated before each run; back up previous results in advance if needed.
- The inference time of each image is printed during testing, and the average total time is reported at the end.
- **The training code will be released after the paper is accepted.**
- If the uploaded code is incomplete due to our oversight, we will supplement it promptly.


## Pretrained Weights

The main pretrained model `model-1-best.pt` is already included in this repository under `model/`.

The other four weights used for our **stability testing** are shared via Baidu Netdisk:

- Link: https://pan.baidu.com/s/1zUuc7ozVTUr1LOOY0fa6BA?pwd=0417
- Extraction code: `0417`

If you want to reproduce our stability experiment, please download these four weights from the link above.

## Dependencies

The main dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```
