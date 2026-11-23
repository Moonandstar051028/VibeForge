"""CPU/controller tests must not create contexts on an optimization GPU."""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
