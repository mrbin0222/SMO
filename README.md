# SMO

**[ECUST](https://www.ecust.edu.cn/)**

Bin Shen, Zhen Gu, Jiale Zhou, Bingyong Yan, Yiquan Fang, Huifeng Wang

[[`Paper`]()] 

## 📌 Code Disclosure Plan

### Objective
The code disclosure plan for this project aims to promote the sharing and exchange of technology, and to encourage more developers to participate in the development and refinement of the project. We believe that through open-source, we can accelerate the maturity of the project, improve code quality, and spark innovative solutions.

### Scope of Disclosure

Source Code: All source code of the project will be fully disclosed, including core functional modules, utility classes, sample code, etc.

Documentation: Detailed development documentation, user manuals, and API documentation are provided to help users and developers better understand and use this project.

### Disclosure Strategy
Initial Review: Before the code is disclosed, we will conduct an internal code review to ensure that the code meets quality standards and does not contain sensitive information.

Continuous Updates: The project will be continuously maintained and updated, with all updates being publicly available through GitHub.

Community Feedback: We highly value community feedback and encourage users and developers to submit questions, suggestions, and improvement proposals.

### 🗓 Timeline
| Timeline       | Milestone                     |
|----------------|-------------------------------|
| Apr 2025       | Dataset Disclosure|
| May 2025       | Operating Environment Disclosure         |
| Aug 2025       | Function Code Open Source|
| Sep 2025       | Demo Release  |

### 📋 Todo List
- [ ] Dataset Disclosure
- [ ] Operating Environment Disclosure
- [ ] Function Code Open Source
- [ ] Demo Release

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


## 🛠 Installation

### 📦 Requirements (comming soon)
```bash
# requirements.txt
torch>=1.7.0
torchvision>=0.8.0
...
```


### ⚙️ Conda Environment (comming soon)
```bash
conda create -n smo python=3.8 -y
conda activate smo
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
...
```

## 🧩 Core Modules

### 🎯 Segmentation Fundamental Module (comming soon)
Any segmentation model can be used, such as SAM (Segment Anything Model), etc.

Install Segment Anything:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
```

or clone the repository locally and install with

```
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

#### Model Checkpoints

Three model versions of the model are available with different backbone sizes. These models can be instantiated by running

```
from segment_anything import sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

#### Getting Start with SAM

```python
# get masks from a given prompt
from segment_anything import SamPredictor

def init_sam(model_type: str = "vit_h"):
    """Initialize SAM model"""
    predictor = SamPredictor(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
    return predictor

predictor = init_sam()
predictor.set_image("demo.png")
masks, _, _ = predictor.predict(<input_prompts>)

# or generate masks for an entire image:
from segment_anything import SamAutomaticMaskGenerator

def init_sam(model_type: str = "vit_h"):
    """Initialize SAM model"""
    generator = SamAutomaticMaskGenerator(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
    return generator

generator = init_sam()
masks = generator.generate("demo.png")
```

### 🌐 Multimodal Prompter Module (comming soon)
Any text-image alignment model can be used, such as CLIP (Contrastive Language-Image Pre-Training), etc.

```python
import clip

def init_clip(model_name: str = "ViT-B/32"):
    """Initialize CLIP model"""
    model, preprocess = clip.load(model_name)
    return model, preprocess
...
# Alternative models: ALIGN, Florence, OpenCLIP
```

## 🚀 Demos

### 🔍 Cell Segmentation Example (comming soon)
```python

```


## 📜 License
Apache 2.0 License

## Citing SMO (Under Review)

If you use SMO in your research, please cite our paper:

**APA Style**  


**BibTeX**  



Please also consider citing the foundational works that our project builds upon:
- Segment Anything Model (SAM) [[Paper](https://arxiv.org/abs/2304.02643)]
- CLIP [[Paper](https://arxiv.org/abs/2103.00020)]





