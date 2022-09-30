# -*- coding:utf-8 -*-
"""
@author: W4yne
@file: log.py
@time: 2021/11/12 0012
"""
import logging

def get_logger(file_path):
    """ Make python logger """
    # create logger
    logger = logging.getLogger('PL-NAS_Training')
    # create formatter
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger