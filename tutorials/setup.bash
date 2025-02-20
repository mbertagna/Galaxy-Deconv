#!/bin/bash
set -e

python3.11 -m venv Galaxy-Deconv.env
source Galaxy-Deconv.env/bin/activate
pip install -r requirements.txt
pip install opencv-python-headless pandas
galsim_download_cosmos -s 23.5
