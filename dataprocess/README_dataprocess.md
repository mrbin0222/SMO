# Datasets used in MPM fine-tuning

- Trained CLIP using BBBC: BBBC004, BBBC009, BBBC014, BBBC024, BBBC038, BBBC045
- Trained pix2pixHD using part of the BBBC021 dataset to generate new synthetic MCF-7 cells
- Synplex was used to generate synthetic multiplexed histological images

## Example BBBC

### Dataset Processing

For all datasets from the BBBC series with available annotations, and as described in Section III.A, each labeled single-cell object was individually cropped based on mask information.

### Text Description Structure

The text descriptions followed a structured template that included the following information:

- **Cell Type**
- **Microscopy Modality**
- **Channel**
- **Pixel Area Size**
- **...**

### Demonstration Script

Please refer to the `examplebbbc021.ipynb` notebook for the demonstration script.

## Example Synplex

### Modification Overview

We have modified Synplex to enable the direct simulation of individual cellular objects.

### Technical Implementation

The original Synplex was developed in MATLAB. We translated it into Python; however, certain portions of the code perform more optimally in MATLAB. Consequently, we developed a Flask API program to facilitate the transfer of data between MATLAB and Python.

### Installation and Usage

1. **MATLAB Installation:** Install MATLAB (version R2024a is recommended).
2. **Starting the Flask API:** Launch the Flask API using the following command:

    ```bash
    python matlabapi.py
    ```

    The API will be running at `127.0.0.1:8080`.

3. **Generating Single-Cell Images:** Utilize `modelsingle.py` to generate single-cell images. The parameters are configured in the `pythonparameters.yaml` file.

## Citing

- Synplex [[Paper](https://doi.org/10.1109/TMI.2023.3273950)]