"""
    Train a ffn model. 
"""

from pred_ffn.train import train
import logging
import time

if __name__ == "__main__":
    start = time.time()
    train.train_model()
    end = time.time()
    logging.info(f"The model training takes {end-start} seconds. ")