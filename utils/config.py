
import json
import yaml
import argparse
from easydict import EasyDict

def get_config_from_yaml(yaml_file):
    """
    Get the config from a yaml file

    :param string yaml_file: yaml configuration file
    :return: EasyDict config
    """
    with open(yaml_file) as fp:
        config_dict = yaml.safe_load(fp)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)
    return config


def get_args():
    """
    Create argparser for frequent configurations.

    :return: argparser object
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default=None,
        help='The Configuration file')

    argparser.add_argument(
        '-s', '--seed',
        default=100,
        type=int,
        help='The random seed')
    argparser.add_argument(
        '--n_gpu',
        default=1,
        type=int,
        help='Number of GPUs')
    argparser.add_argument(
        '--temp',
        default=-1,
        type=float,
        help='temperature')
    argparser.add_argument(
        '--lr',
        default=-1,
        type=float,
        help='learning rate')

    argparser.add_argument(
        '--score_func',
        default=None,
        help='The path to ckpt')
    argparser.add_argument(
        '--backbone',
        default=None,
        help='The path to ckpt')
    argparser.add_argument(
        '--ckpt',
        default=None,
        help='The path to ckpt')
    argparser.add_argument(
        '--tag',
        default=None,
        help='additional information')
    argparser.add_argument(
        '--vlm',
        default=None,
        help='choose VLM')
    args = argparser.parse_args()
    return args

def get_config():
    """
    Create experimental config from argparse and config file.

    :return: Configuration EasyDict
    """
    # read manual args
    args = get_args()
    config_file = args.config

    # load experimental configuration
    if config_file.endswith('json'):
        config = get_config_from_json(config_file)
    elif config_file.endswith('yaml'):
        config = get_config_from_yaml(config_file)
    else:
        raise Exception("Only .json and .yaml are supported!")

    # reset config from args

    config.seed = args.seed
    config.n_gpu = args.n_gpu
    if args.tag:
        config.tag = args.tag

    if args.score_func:
        config.score_func = args.score_func
    if args.temp > 0:
        config.temp = args.temp
    if args.lr > 0:
        config.lr = args.lr
    if args.vlm:
        config.vlm = args.vlm
    if args.backbone:
        config.backbone = args.backbone
    if args.ckpt:
        config.ckpt = args.ckpt
    config.config_file = args.config
    return config

