# INSTALL DEPENDENCIES
python3.11 -m venv Galaxy-Deconv.env
source Galaxy-Deconv.env/bin/activate
pip install -r requirements.txt

# GENERATE DATASET
galsim_download_cosmos -s 23.5
python generate_data.py --task Deconv --n_train 40000

# TRAIN BASELINE MODEL
time { python train.py --model Unrolled_ADMM --n_iters 2 --n_epochs 10 --loss MultiScale --lr 1e-4; }