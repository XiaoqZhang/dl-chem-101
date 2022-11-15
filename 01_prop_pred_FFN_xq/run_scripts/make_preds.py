"""
    Make predictions. 
"""

from pred_ffn.predict import predict
import time
import logging

if __name__ == "__main__":
    start = time.time()
    predict.predict()
    end = time.time()
    logging.info(f"Prediction takes {end-start} seconds. ")