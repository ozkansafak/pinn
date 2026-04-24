#!/bin/bash
set -e

python train.py --lid uniform --network 1x --epochs 30000
python train.py --lid uniform --network 2x --epochs 30000
python train.py --lid sigmoid --network 1x --epochs 30000
python train.py --lid sigmoid --network 2x --epochs 30000
