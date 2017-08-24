# Hermes Bib Detection

This script detects bibs for images in a given directory. These images can
either be original raw images or cropped images using the person detection
[script](https://github.com/alexcu/hermes-training-utils/blob/master/person_detect.rb).

## Installing

Use the `setup.sh` script to install.

## Usage

Run the scripts in the following order:

1. `person_detect.rb` (optional) to detect and crop people
2. `detect.py` with bibs
6. `person_aggregate.py` (optional) to aggregate results from cropped people
4. `detect.py` with text
5. `recognise.rb` to recognise the text

## Pipeline

To run the pipeline:

```
make run \
  JOB_ID=unique_job_id
  IN_DIR=/path/to/input/images \
  OUT_DIR=/path/to/output \
  DARKNET_DIR=/path/to/darknet \
  PICKLE_CONFIG_BIB=/path/to/bib/models/config.pickle \
  PICKLE_CONFIG_TXT=/path/to/text/models/config.pickle \
  TESSERACT_BIN_DIR=/path/to/tesseract \
  TESSDATA_DIR=/path/to/tessdata \
  CROP_PEOPLE=1
```

Use `CROP_PEOPLE=0` to *not* crop people using YOLO.

## Output

Sample annotated image:

![Sample Output](https://i.imgur.com/5Cazpj2.png)

Each annotation on the image is in the format `string_read [bib_accuracy% / txt_accuracy %]`.

Respective JSON file for this image:

```json
{
   "char":{
      "regions":[
         {
            "y2":641,
            "width":62,
            "height":37,
            "char":"~",
            "x2":477,
            "y1":567,
            "x1":353
         }
      ],
      "string":"~",
      "elappsed_seconds":0.338908
   },
   "person":{
      "regions":[
         {
            "y1":305,
            "x2":488,
            "x1":152,
            "y2":813,
            "accuracy":0.68619
         },
         {
            "y1":256,
            "x2":829,
            "x1":285,
            "y2":863,
            "accuracy":0.676243
         },
         {
            "y1":246,
            "x2":1016,
            "x1":615,
            "y2":841,
            "accuracy":0.659793
         }
      ],
      "elapsed_seconds":1.251645
   },
   "bib":{
      "regions":[
         {
            "y1":403,
            "x2":874,
            "x1":705,
            "y2":540,
            "accuracy":0.9993394017219543
         },
         {
            "y1":523,
            "x2":452,
            "x1":312,
            "y2":641,
            "accuracy":0.9995879530906677
         },
         {
            "y2":453,
            "height":36,
            "x2":470,
            "width":36,
            "y1":407,
            "x1":424,
            "accuracy":0.8520742654800415
         },
         {
            "y2":384,
            "height":123,
            "x2":699,
            "width":150,
            "y1":256,
            "x1":511,
            "accuracy":0.8540436625480652
         },
         {
            "y2":733,
            "height":164,
            "x2":666,
            "width":218,
            "y1":515,
            "x1":394,
            "accuracy":0.9998284578323364
         }
      ],
      "elapsed_seconds":29.776029348373413
   },
   "text":{
      "regions":[
         {
            "y2":604,
            "y1":567,
            "height":25,
            "width":50,
            "x2":415,
            "x1":353,
            "accuracy":0.9343273043632507
         }
      ],
      "elapsed_seconds":9.258251905441284
   }
}
```

The `elapsed_time` field is in **seconds**.
