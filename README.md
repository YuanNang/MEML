
# 2D Materials Prediction Pipeline

This pipeline predicts the **stability**, **electronic conductivity**, and **bandgap types** of 2D materials based on crystal and X-ray diffraction (XRD) data.

## Steps to Use

### 1. Generate Descriptors for Training
Navigate to the `descriptor` directory and run the following to generate the necessary descriptors:

```bash
cd descriptor
unzip ../data/cifs.zip -d ../data
python descriptor.py
```

### 2. Train the Model
Go to the `model` directory and train the model:

```bash
cd model
python train.py
```

### 3. Prepare Datasets for MP (Materials Project)
In the `MP` directory, extract the CIF files and run the scripts to prepare the data:

```bash
cd MP
unzip mp_cif.zip
python get_mp_datasets.py
python get_mp_XRD.py  # This may take some time
```

### 4. Predict Properties
Use the trained model to predictproperties f or the prepared dataset:

```bash
python mp_predict.py
```

## Requirements

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- `get_mp_XRD.py` may take a while to complete due to the complexity of the calculations.
