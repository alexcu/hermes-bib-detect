#!/usr/bin/env bash

EVAL_SET=$1 # Evaluation set is either `I' for ideal or `R' for realistic
TRAIN_NO=$2 # Train number is either `1', `100', `500', or `all'
CROP_HUM=$3 # Crop on humans is either `0' or `1'
CMD=$4 # Command to run (default to `run')

if [ "$EVAL_SET" == "i" ]; then
  EVAL_SET="I"
  IN_DIR="/home/alex/data/bibs/eval/1"
elif [ "$EVAL_SET" == "r" ]; then
  EVAL_SET="R"
  IN_DIR="/home/alex/data/bibs/eval/2_or_more"
else
  echo "Bad eval set"
  exit 1;
fi

if [ "$TRAIN_NO" == "all" ]; then
  TRAIN_NO="ALL"
  PICKLE_CONFIG_BIB="/home/alex/out/models/bibs/02-08-2017/config.pickle"
else
  PICKLE_CONFIG_BIB="/home/alex/out/models/bibs/31-08-2017/train_$TRAIN_NO/config.pickle"
fi

if [ "$CROP_HUM" == "1" ]; then
  CROP_HUM_STR="CR"
else
  CROP_HUM_STR="NC"
fi

# Actual params
JOB_ID="$EVAL_SET-$TRAIN_NO-$CROP_HUM_STR"
OUT_DIR="/home/alex/out/eval/$(date +%d-%m-%Y)"
DARKNET_DIR="/home/alex/bin/hermes-bib-detect/bin/darknet"
PICKLE_CONFIG_TXT="/home/alex/out/models/text/18-08-2017/config.pickle"
TESSERACT_BIN_DIR="/home/alex/local/bin"
CROP_PEOPLE=$CROP_HUM

echo "Evaluating set: $JOB_ID..."
echo " - JOB_ID=$JOB_ID"
echo " - IN_DIR=$IN_DIR"
echo " - OUT_DIR=$OUT_DIR"
echo " - DARKNET_DIR=$DARKNET_DIR"
echo " - PICKLE_CONFIG_BIB=$PICKLE_CONFIG_BIB"
echo " - PICKLE_CONFIG_TXT=$PICKLE_CONFIG_TXT"
echo " - TESSERACT_BIN_DIR=$TESSERACT_BIN_DIR"
echo " - CROP_PEOPLE=$CROP_PEOPLE"

if [ -z "$CMD" ]; then
  CMD="run"
fi

make $CMD \
  JOB_ID=$JOB_ID\
  IN_DIR=$IN_DIR\
  OUT_DIR=$OUT_DIR\
  DARKNET_DIR=$DARKNET_DIR\
  PICKLE_CONFIG_BIB=$PICKLE_CONFIG_BIB\
  PICKLE_CONFIG_TXT=$PICKLE_CONFIG_TXT\
  TESSERACT_BIN_DIR=$TESSERACT_BIN_DIR\
  CROP_PEOPLE=$CROP_PEOPLE\
