# Hermes Bib Detection

This script detects bibs for images in a given directory. These images can
either be original raw images or cropped images using the person detection
[script](https://github.com/alexcu/hermes-training-utils/blob/master/person_detect.rb).

## Usage

You will need the following:

1. The training configuration file (`config.pickle`) along with the model hdf5
   file. These should sit in the same directory.
2. Input files in a directory.

To run:

```bash
$ python bib_detect.py -c /path/to/out/models/config.pickle \
                       -i /path/to/input/images \
                       -o /path/to/output
                       [-g][-j]
```

The `-g` and `-j` switches are optional:

- `-g` will spit out _only_ annotated images.
- `-j` will spit out _only_ JSON files.

Not specifying either will spit out both to the output directory.

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
      ]
   },
   "elapsed_time":0.3909590244293213
}
```
