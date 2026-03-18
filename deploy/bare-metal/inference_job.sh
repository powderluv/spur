#!/bin/bash
# Spur job wrapper: activate venv and run the TP inference test.
# Deployed to ~/spur/ on each cluster node.
source ~/spur/venv/bin/activate
exec python3 ~/spur/inference_test.py
