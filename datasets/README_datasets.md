# Datasets used in segmentation test
we employed four publicly accessible datasets and one private dataset.
## 🔬 Publicly Accessible Datasets
| Dataset                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| LIVECell                 | A substantial and high-quality cell dataset including a diverse range of cell morphologies. This dataset consists of 5,239 microscope images and a total of 1,686,352 labeled cells, which were manually annotated and verified by experts.        |
| Cell Tracking Challenge  | A classical cell/nucleus dataset. Seven categories in the Cell Tracking Challenge dataset were utilized: DIC-C2DH-HeLa, Fluo-N2DL-HeLa, Fluo-N2DH-GOWT1, Fluo-N3DH-CHO, Fluo-N2DH-SIM, PhC-C2DH-U373, and PhC-C2DL-PSC. |
| MoNuSeg                  | A nucleus dataset comprising 30 images for training, annotated with 21,623 cellular instances, along with 14 images designated for testing, featuring 6,697 cell annotations. Each image within this dataset was captured from H&E-stained tissue specimens at 40x optical magnification. |
| TNBC                     | A nucleus dataset containing H&E-stained pathology images of 11 triple-negative breast cancer (TNBC) patients, with 40x magnification. Among the 50 images, 4022 nuclei were labeled. |

## 🔬 Private Dataset

### datasets/B3/

Share with google drive:
[https://drive.google.com/drive/folders/1wtE1G8aXoCPOrdWgaCm-CUQzm2AeilEx?usp=sharing]

H1299 and HeLa cells, both derived from human tissues, were cultured at densities varying between 20% and 80%. Imaging was performed with an Olympus LUCPLFLN 20X objective, capturing three channels (BF, CFP and GFP) at 30-minute intervals across a duration of 27 hours. The ground truth of this dataset has not yet been fully annotated.

# Datasets used in MPM fine-tuning

- Trained CLIP using BBBC: BBBC004, BBBC009, BBBC014, BBBC024, BBBC038, BBBC045
- Trained pix2pixHD using part of the BBBC021 dataset to generate new synthetic MCF-7 cells
- Synplex was used to generate synthetic multiplexed histological images

## Citing

- OpenCLIP [[Paper](https://doi.org/10.1109/CVPR52729.2023.00276)]
- pix2pixHD [[Paper](https://doi.org/10.1109/CVPR.2018.00917)]
- Synplex [[Paper](https://doi.org/10.1109/TMI.2023.3273950)]
