from logging import Logger
import logging

__author__ = 'itay'


class ProjectParams(object):
    k_fold = 3
    logger = Logger("gp_log")
    logger.addHandler(logging.StreamHandler())
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.Formatter = formatter
