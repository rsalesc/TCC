#!/bin/bash

LR=0.0008
MARGIN=0.2
LINE_CAPACITY="64 128"
CHAR_CAPACITY="32 64"
DROPOUT_FC=0.0
DROPOUT_CHAR=0.0
DROPOUT_LINE=0.0
DROPOUT_INTER=0.0
MAX_LINES=100
NAME=tcc-huge

CH_CAP=${CHAR_CAPACITY//[[:space:]]/\-}
LINE_CAP=${LINE_CAPACITY//[[:space:]]/\-}
echo "$CH_CAP"

FULL_NAME=${NAME}/${LR}_${MARGIN}_dchar${DROPOUT_CHAR}_dline${DROPOUT_LINE}/lines${MAX_LINES}_cchar_${CH_CAP}_cline${LINE_CAP}
DIR=/opt/tensorboard/lstm/triplet/$FULL_NAME

if [[ $# -eq 0 ]]; then
        read -p "Are you sure? " -n 1 -r
        echo    # (optional) move to a new line
        if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Removing $DIR..."
                rm -rf $DIR
        fi
        exit 0
fi

python3 nn.py --name ${FULL_NAME} \
	--training-file .cache/training.pkl \
	--validation-file .cache/validation.pkl \
	--embedding-file .cache/validation.pkl \
        --validation-batch-size 64 \
        --lr ${LR} \
        --eval-every 100 \
        --max-epochs 256 \
        --no-checkpoint \
        lstm \
        --char-capacity ${CHAR_CAPACITY} \
        --line-capacity ${LINE_CAPACITY} \
        --dropout-char ${DROPOUT_CHAR} \
        --dropout-line ${DROPOUT_LINE} \
        --dropout-fc ${DROPOUT_FC} \
        --dropout-inter ${DROPOUT_INTER} \
        --max-lines ${MAX_LINES} \
        triplet \
        --margin ${MARGIN}
