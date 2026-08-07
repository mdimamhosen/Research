# Kaggle — Bangla OCR (PaddleOCR-VL on free GPU)

Copy-paste these cells in order. **No need to copy PDFs one-by-one.**

## 0. Accelerator

Notebook → **Accelerator** → **GPU T4** (or P100).

## 1. Clone / enter project

```python
!git clone https://github.com/mdimamhosen/ResearchWorkspace.git
%cd /kaggle/working/ResearchWorkspace/EDGE/BanglaDialectDataset
!ls
```

If you already cloned, only run the `%cd` cell.

## 2. Install (Paddle GPU + OCR extras)

`paddlepaddle-gpu` alone is **not** enough — you also need `paddleocr` + `paddlex[ocr]`.

Ignore unrelated pip conflict warnings (`google-adk`, `ydata-profiling`) — they are Kaggle preinstall noise.

```python
!pip install -q paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
!pip install -q "paddleocr>=3.0.0" "paddlex[ocr]>=3.7.0" pymupdf Pillow python-dotenv tqdm numpy

import paddle
from paddleocr import PaddleOCRVL
print("paddle", paddle.__version__, "cuda", paddle.is_compiled_with_cuda(), "device", paddle.device.get_device())
print("PaddleOCRVL OK")
```

You want: `cuda True` and `device gpu:0` and `PaddleOCRVL OK`.

## 3. Run on ALL dataset PDFs (no copy)

Attach your PDF dataset(s) in the right sidebar (**Add input**). Then:

```python
import os
os.environ["OCR_ENGINE"] = "paddle"
os.environ["PADDLE_OCR_BACKEND"] = "vl"
os.environ["PADDLE_OCR_DEVICE"] = "gpu:0"
os.environ["PADDLE_OCR_USE_LAYOUT"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Finds every .pdf under /kaggle/input automatically
!python cli.py --kaggle -o /kaggle/working/txt --engine paddle --skip-existing
```

Or point at one dataset folder:

```python
!python cli.py -i /kaggle/input/datasets/mdimamhosen/testpdf1 -o /kaggle/working/txt -r --engine paddle --force
```

Smoke-test 2 pages first:

```python
!python cli.py -i /kaggle/input/datasets/mdimamhosen/testpdf1 -o /kaggle/working/txt -r --engine paddle --page-start 1 --page-end 2 --force
```

## 4. Download results

```python
!ls -la /kaggle/working/txt/
!cd /kaggle/working/txt && zip -r /kaggle/working/bangla_txt_out.zip .
```

Download `/kaggle/working/bangla_txt_out.zip` from the file browser.

## Why your last error happened

```text
paddle 3.2.2 True gpu:0     ← GPU Paddle framework OK
Failed ... PaddleOCR is not installed (or VL extras missing)
```

You installed **PaddlePaddle**, but not **PaddleOCR-VL** packages. Cell 2 fixes that.

Pip messages about `google-adk` / `ydata-profiling` / `ccache` are **not** fatal for this project.
