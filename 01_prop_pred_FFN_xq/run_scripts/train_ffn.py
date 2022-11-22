"""
    Train a ffn model. 
"""

import pred_ffn.train as train
import logging
import time

if __name__ == "__main__":
    start = time.time()
    train.train_model()
    end = time.time()
    logging.info(f"The model training takes {end-start} seconds. ")