#!/usr/bin/env python3

from deeplodocus.brain import Brain
import argparse


def main(args):
    config = args.c

    brain = Brain(config)
    brain.wake()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default="config/config_depthnet.yaml",
                        help="Path to the config yaml file")
    arguments = parser.parse_args()
    main(arguments)
