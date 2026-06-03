# OLDA-Bench


**An Interpretable AI System for Oral Leukoplakia Progression: From Early Screening to Lesion Delineation**

---

## Overview

Oral leukoplakia is one of the most common oral potentially malignant disorders (OPMDs) and represents a critical precursor to oral cancer. However, current diagnostic workflows largely rely on invasive biopsy procedures and subjective clinical assessment, limiting their suitability for large-scale screening and longitudinal monitoring.

To address these challenges, we introduce **OLDA-Bench (Oral Leukoplakia Diagnosis and Analysis Benchmark)**, the first benchmark specifically designed for oral leukoplakia progression recognition and interpretable lesion assessment


## 📦 Dataset Structure

Images identified as Leukoplakia or Leukoplakia with Cancer in the first stage are passed to the segmentation network. This stage uses a dataset in the standard COCO format, structured as follows:
```
./root_data/
│
├── train/
│ ├── xxx.jpg
│ ├── ...
├── val/
│ ├── xxx.jpeg
│ ├── ...
├── annotations
│ ├── train.json
│ ├── val.json
```
The instance segmentation task includes two categories:

  - **Leukoplakia**
  - **Leukoplakia Cancer**

The .json annotation files provide pixel-level segmentation masks and category labels for each lesion instance.


---

## 📊 Dataset Split

- **Training set**: 389 images  
- **Validation set**: 44 images  
---


## 🧪 Benchmark with MMdetection3.x & Detection2

### 🔹 Training

```bash
# Single-GPU training
python tools/train.py\
  work_dirs_mask2former_swin_b/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./tools/train.py\
  work_dirs_mask2former_swin_b/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py\
  --launcher pytorch

```
You can replace config with any supported architecture name from mmdetection3.x.

### 🔹 Evaluation
After training, evaluate a model checkpoint on the test set:

```bash
# Single-GPU evaluation
python tools/test.py\
  work_dirs_sparseinsts/sparseinst_r50_iam_8xb8-ms-270k_coco.py\
  work_dirs_sparseinsts/best_coco_segm_mAP_iter_8500.pth

# Multi-GPU evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/test.py \
    work_dirs_sparseinsts/sparseinst_r50_iam_8xb8-ms-270k_coco.py \
    work_dirs_sparseinsts/best_coco_segm_mAP_iter_8500.pth \
    --launcher pytorch 

```


### 🔹Additional Info
The OLDA-Bench is based on our previous work (environment code base) published in [*IEEE JBHI CDTM*](https://github.com/qklee-lz/CDTM) and [*OLPR*](https://github.com/qklee-lz/OLPR/).
