import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)   # my_sft/trainer.py
sys.path.insert(1, ROOT)   # probe_crm.SYSTEM, data/
