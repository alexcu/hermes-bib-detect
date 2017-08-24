#!/usr/bin/env bash

SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

check_pip3_dependency() {
  if [ $( pip3 list | grep "$1" | wc -l ) -eq 0 ]; then
    return 1
  else
    return 0
  fi
}

check_system_dependency() {
  if [ $( which $1 | wc -l ) -eq 0 ]; then
    return 1
  else
    return 0
  fi
}

check_lib_dependency() {
  if [ $( dpkg-query -l "$1" | grep "$1" | wc -l ) -ne 1 ]; then
    return 1
  else
    return 0
  fi
}

check_ruby_dependency() {
  if [ $( gem list | grep "$1" | wc -l ) -eq 0 ]; then
    return 1
  else
    return 0
  fi
}

check_darknet_installed() {
  if [ -d "$SRC_DIR/bin/darknet" ]; then
    return 0
  else
    return 1
  fi
}

announce_success() {
  echo "Success!"
  echo "  - Your DARKNET_DIR is: $SRC_DIR/bin/darknet"
  echo "  - Your TESSERACT_BIN_DIR is: $(dirname $(which tesseract))"
}

install_git() {
  sudo apt-get install -y git
}

install_tesseract() {
  sudo apt-get install -y tesseract-ocr
}

install_darknet() {
  mkdir $SRC_DIR/bin
  git clone https://github.com/alexcu/darknet.git bin/darknet
  make bin/darknet
  wget -P bin/darknet https://pjreddie.com/media/files/tiny-yolo-voc.weights
}

install_python3() {
  sudo apt-get install -y python3-pip python3-dev
  pip3 install --upgrade pip
}

install_opencv() {
  pip3 install opencv-python
}

install_keras() {
  pip3 install keras
}

install_tensorflow() {
  pip3 install tensorflow
}

install_h5py() {
  pip3 install h5py
}

install_ruby() {
  sudo apt-get install -y ruby ruby-all-dev
}

install_imagemagick() {
  sudo apt-get install -y imagemagick libmagickcore-dev libmagickwand-dev
}

install_rmagick() {
  sudo gem install rmagick
}

check_system_dependency "git"               || install_git         &&
check_system_dependency "tesseract"         || install_tesseract   &&
check_darknet_installed                     || install_darknet     &&
check_system_dependency "python3"           || install_python3     &&
check_system_dependency "pip3"              || install_python3     &&
check_pip3_dependency   "opencv-python"     || install_opencv      &&
check_pip3_dependency   "Keras"             || install_keras       &&
check_pip3_dependency   "tensorflow"        || install_tensorflow  &&
check_pip3_dependency   "h5py"              || install_h5py        &&
check_system_dependency "ruby"              || install_ruby        &&
check_system_dependency "convert"           || install_imagemagick &&
check_lib_dependency    "libmagickcore-dev" || install_imagemagick &&
check_lib_dependency    "libmagickwand-dev" || install_imagemagick &&
check_ruby_dependency   "rmagick"           || install_rmagick     &&
announce_success
