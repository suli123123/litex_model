#!/bin/bash

accelerate launch \
    --config_file hardware_config.yaml \
    train.py