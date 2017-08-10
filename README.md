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

```
$ python bib_detect.py -c /path/to/out/models/config.pickle \
                       -i /path/to/input/images \
                       -o /path/to/output
                       [-g][-j]
```

The `-g` and `-j` switches are optional:

- `-g` will spit out _only_ annotated images.
- `-j` will spit out _only_ JSON files.

Not specifying either will spit out both to the output directory.

