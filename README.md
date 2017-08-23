# Hermes Bib Detection

This script detects bibs for images in a given directory. These images can
either be original raw images or cropped images using the person detection
[script](https://github.com/alexcu/hermes-training-utils/blob/master/person_detect.rb).

## Usage

Run the scripts in the following order:

1. `person_detect.rb` (optional) to detect and crop people
2. `detect.py` with bibs
6. `person_aggregate.py` (optional) to aggregate results from cropped people
4. `detect.py` with text
5. `recognise.rb` to recognise the text

### `detect.py`

You will need the following:

1. The training configuration file (`config.pickle`) along with the model hdf5
   file. These should sit in the same directory.
2. Input files in a directory.

**It is imperative that you name the model `model_bib_[date]_frcnn.hdf5` for
bib models and similarly `model_text_[date]_frcnn.hdf5` for text models.**

To run:

```bash
$ python detect.py -c /path/to/out/models/config.pickle \
                   -i /path/to/input/images \
                   -o /path/to/output
```

To run on

## Output

Sample annotated image:

![Sample Output](https://i.imgur.com/gFpCPCC.jpg)

Respective JSON file for this image:

```json
{
   "bib":{
      "regions":[
         {
            "y1":389,
            "x2":272,
            "x1":136,
            "y2":486,
            "width": 136,
            "height": 97,
            "accuracy":0.9993014335632324
         },
         {
            "y1":486,
            "x2":584,
            "x1":467,
            "y2":584,
            "width": 117,
            "height": 228,
            "accuracy":0.9964402318000793
         }
      ],
      "elapsed_seconds":0.3909590244293213
   }
}
```

The `elapsed_time` field is in **seconds**.
