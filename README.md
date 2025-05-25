# SMO

**[ECUST](https://www.ecust.edu.cn/)**

Bin Shen, Zhen Gu, Jiale Zhou, Bingyong Yan, Yiquan Fang, Huifeng Wang

[[`Paper`]()] 

## 📌 Code Disclosure Plan

### 🗓 Timeline
| Timeline       | Milestone                     |
|----------------|-------------------------------|
| Apr 2025       | Dataset Disclosure|
| May 2025       | Function Code Open Source|
| May 2025       | Demo Release  |

### 📋 Todo List
- [x] Dataset Disclosure
- [x] Function Code Open Source
- [x] Demo Release Part 1
- [ ] Demo Release Part 2

## 📁 Datasets
we employed four publicly accessible datasets and one private dataset.
### 🔬 Publicly Accessible Datasets
| Dataset                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| LIVECell                 | A substantial and high-quality cell dataset including a diverse range of cell morphologies. This dataset consists of 5,239 microscope images and a total of 1,686,352 labeled cells, which were manually annotated and verified by experts.        |
| Cell Tracking Challenge  | A classical cell/nucleus dataset. Seven categories in the Cell Tracking Challenge dataset were utilized: DIC-C2DH-HeLa, Fluo-N2DL-HeLa, Fluo-N2DH-GOWT1, Fluo-N3DH-CHO, Fluo-N2DH-SIM, PhC-C2DH-U373, and PhC-C2DL-PSC. |
| MoNuSeg                  | A nucleus dataset comprising 30 images for training, annotated with 21,623 cellular instances, along with 14 images designated for testing, featuring 6,697 cell annotations. Each image within this dataset was captured from H&E-stained tissue specimens at 40x optical magnification. |
| TNBC                     | A nucleus dataset containing H&E-stained pathology images of 11 triple-negative breast cancer (TNBC) patients, with 40x magnification. Among the 50 images, 4022 nuclei were labeled. |

### 🔬 Private Dataset

H1299 and HeLa cells, both derived from human tissues, were cultured at densities varying between 20% and 80%. Imaging was performed with an Olympus LUCPLFLN 20X objective, capturing three channels (BF, CFP and GFP) at 30-minute intervals across a duration of 27 hours. The ground truth of this dataset has not yet been fully annotated.


## 🧩 Core Modules

- Cell Segmentation Based on Text Prompts (comming soon)
- Nucleus Segmentation Based on Layout Prompts and Text Prompts (completed)
- Cell Division Event Recognition Based on Feature Point Prompts and Text Prompts
- Synthetic Data Generation and Multimodal Prompter Module Fine-Tuning (completed)

## 🚀 Demos

### Nucleus Segmentation Based on Layout Prompts and Text Prompts

[See README_layoutandtext](layoutandtext/README_layoutandtext.md)

### Synthetic Data Generation and Multimodal Prompter Module Fine-Tuning

[See README_datasets](datasets/README_datasets.md)
[See README_dataprocess](dataprocess/README_dataprocess.md)

## 📜 License

Apache 2.0 License

## Citing SMO (Under Review)

If you use SMO in your research, please cite our paper:

**APA Style**  


**BibTeX**  

