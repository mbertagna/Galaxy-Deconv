# Getting Started  

Follow these steps to set up your environment and install dependencies for this project.  

## 1. Clone the Repository  

```bash
git clone https://github.com/mbertagna/Galaxy-Deconv.git
cd Galaxy-Deconv
```

## 2. Create a Virtual Environment  

```bash
python3.11 -m venv Galaxy-Deconv.env
```

## 3. Activate the Virtual Environment

```bash
source Galaxy-Deconv.env/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

You can now run the project demo [final_report.ipynb](https://github.com/mbertagna/Galaxy-Deconv/blob/main/final_report.ipynb).

---

# Generating a Simulated Dataset

Follow these steps to generate the simulated dataset used for this project.

## 1. Download COSMOS

```bash
galsim_download_cosmos -s 23.5
```

## 2. Generate Training and Validation Datasets

```bash
python generate_data.py --task Deconv --n_train 40000
```

Note: You may need to update paths within the generate_data.py file.

---

# Training a Model

## Run train.py

```bash
{ time python train.py --model Unrolled_ADMM --n_iters <number_of_admm_iters> --n_epochs <num_epochs> --loss <loss_function> --lr <learning_rate>; } 2>&1 | tee "train_output_$(date +'%Y%m%d_%H%M%S').txt"
```

Adjustable Parameters:
- --n_iters: Number of iterations in the unrolled ADMM network
- --n_epochs: Number of training epochs
- --loss: Loss function, choose from:
  - MultiScale
  - BestEllipse
  - MomentBasedLoss
  - ShapeletMomentsLoss
- --lr: Learning rate

This command also logs the training loss and runtime in an output file with a timestamped filename.

Example:
```bash
{ time python train.py --model Unrolled_ADMM --n_iters 2 --n_epochs 10 --loss MultiScale --lr 1e-4; } 2>&1 | tee "train_output_$(date +'%Y%m%d_%H%M%S').txt"
```

Note: You may need to update paths within the train.py, utils/utils_train.py, and utils/utils_data.py files.

The trained model will be saved in the ./saved_models_shape_loss/ directory, along with a graph showing the loss over epochs. Inference results can be viewed in tutorials/deconv.ipynb.

---
