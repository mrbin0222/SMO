# A Deep Learning Multimodal Fusion-Based Method for Cell and Nucleus Segmentation

Bin Shen, Zhen Gu, Jiale Zhou, Bingyong Yan, Yiquan Fang, Huifeng Wang

[![IEEE-Xplore](https://img.shields.io/badge/IEEE_Xplore-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/document/11096725) 

Accepted for publication in [IEEE Transactions on Medical Imaging] (TMI) 2025.

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
- [x] Demo Release

## 📁 Datasets

we employed four publicly accessible datasets and one private dataset.

See [README_datasets](datasets/README_datasets.md)

## 🧩 Core Modules

- Cell Segmentation Based on Text Prompts
- Nucleus Segmentation Based on Layout Prompts and Text Prompts
- Cell Division Event Recognition Based on Feature Point Prompts and Text Prompts
- Synthetic Data Generation and Multimodal Prompter Module Fine-Tuning

## 🚀 Demos

### Cell Segmentation Based on Text Prompts

See [README_textprompts](layoutandtext/README_textprompts.md)

### Nucleus Segmentation Based on Layout Prompts and Text Prompts

See [README_layoutandtext](layoutandtext/README_layoutandtext.md)

### Synthetic Data Generation and Multimodal Prompter Module Fine-Tuning

See [README_datasets](datasets/README_datasets.md)

See [README_dataprocess](dataprocess/README_dataprocess.md)

### Cell Division Event Recognition Based on Feature Point Prompts and Text Prompts

See [README_featureandtext](layoutandtext/README_featureandtext.md)

## 📜 License

Apache 2.0 License

## Citing SMO

If you use SMO in your research, please cite our paper:

```latex
@ARTICLE{11096725,
  author={Shen, Bin and Gu, Zhen and Zhou, Jiale and Yan, Bingyong and Fang, Yiquan and Wang, Huifeng},
  journal={IEEE Transactions on Medical Imaging}, 
  title={A Deep Learning Multimodal Fusion-Based Method for Cell and Nucleus Segmentation}, 
  year={2025},
  keywords={Image segmentation;Computer architecture;Microprocessors;Training;Deep learning;Weak supervision;Visualization;Computational modeling;Transfer learning;Linguistics;Cellular image;deep learning;cell segmentation;nucleus segmentation;multimodal},
  doi={10.1109/TMI.2025.3592625}}
```