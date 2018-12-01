#!/bin/bash

CMD="python3 nn.py --training-file .cache/training.pkl --validation-file .cache/validation.pkl --embedding-file .cache/validation.pkl"

$CMD --name test --lr=0.0005 lstm --max-lines=30 --max-chars=30 triplet --margin=0.2
