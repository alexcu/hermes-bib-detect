#
# Makefile for Hermes bib detection
# Alex Cummaudo 23 August 2017
#

check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined $1$(if $2, ($2))))

setup:
	$(info Cloining required dependencies...)
	sudo apt-get install git tesseract-ocr
	git clone https://github.com/alexcu/darknet.git bin/darknet
	make bin/darknet
	wget -P bin/darknet https://pjreddie.com/media/files/tiny-yolo-voc.weights
	$(info Installing python dependencies from pip...)
	sudo apt-get install python3-pip python3-dev
	pip3 install opencv-python keras tensorflow
	$(info Installing ruby & imagemagick dependencies...)
	sudo apt-get install imagemagick libmagickcore-dev libmagickwand-dev ruby-all-dev
	gem install rmagick

# Conditionally configure run based on whether we want to crop people
ifeq ($(CROP_PEOPLE),1)
run: prepare \
     person_detect \
     bib_detect \
     person_aggregate \
     text_detect \
     text_recognise \
     annotate
else
run: prepare \
     bib_detect \
     text_detect \
     text_recognise \
     annotate
endif

prepare:
	$(call check_defined, JOB_ID, job identifier)
	$(call check_defined, IN_DIR, input directory)
	$(call check_defined, OUT_DIR, output directory)
	$(call check_defined, DARKNET_DIR, directory to Darknet)
	$(call check_defined, PICKLE_CONFIG_BIB, bib pickle config file)
	$(call check_defined, PICKLE_CONFIG_TXT, text pickle config file)
	$(call check_defined, TESSERACT_BIN_DIR, directory to Tesseract binary)
	$(call check_defined, TESSDATA_DIR, tessdata directory)
	$(call check_defined, CROP_PEOPLE, whether to crop people -- should be 0 or 1)
	$(info Creating output directory at $(OUT_DIR)...)
	rm -rf $(OUT_DIR)/$(JOB_ID)
	mkdir -p $(OUT_DIR)/$(JOB_ID)
	cp -r $(IN_DIR) $(OUT_DIR)/$(JOB_ID)/input
	$(info Running mogrify on all input images to ensure auto-orientation...)
	mogrify -auto-orient $(OUT_DIR)/$(JOB_ID)/input/*.jpg
# TODO: Copy over if any argus files in there into gt.json

person_detect:
	$(info Running person detection...)
	./person_detect.rb $(OUT_DIR)/$(JOB_ID)/input $(OUT_DIR)/$(JOB_ID)/out/person $(DARKNET_DIR) -c

bib_detect:
	$(info Running bib detection...)
ifeq ($(CROP_PEOPLE),1)
	python3 detect.py -i $(OUT_DIR)/$(JOB_ID)/out/person -o $(OUT_DIR)/$(JOB_ID)/out/bib -c $(PICKLE_CONFIG_BIB) -t bib
else
	python3 detect.py -i $(OUT_DIR)/$(JOB_ID)/input -o $(OUT_DIR)/$(JOB_ID)/out/bib -c $(PICKLE_CONFIG_BIB) -t bib
endif

person_aggregate:
	$(info Running aggregation of cropped people...)
	python3 person_aggregate.py $(OUT_DIR)/$(JOB_ID)/input $(OUT_DIR)/$(JOB_ID)/out/aggregate $(OUT_DIR)/$(JOB_ID)/out/bib $(OUT_DIR)/$(JOB_ID)/out/person

text_detect:
	$(info Running text detection...)
ifeq ($(CROP_PEOPLE),1)
	python3 detect.py -i $(OUT_DIR)/$(JOB_ID)/out/aggregate -o$(OUT_DIR)/$(JOB_ID)/out/text -c $(PICKLE_CONFIG_TXT) -t text
else
	python3 detect.py -i $(OUT_DIR)/$(JOB_ID)/out/bib -o$(OUT_DIR)/$(JOB_ID)/out/text -c $(PICKLE_CONFIG_TXT) -t text
endif

text_recognise:
	$(info Running text recognition...)
	./recognise.rb $(OUT_DIR)/$(JOB_ID)/out/text $(OUT_DIR)/$(JOB_ID)/out/chars $(TESSERACT_BIN_DIR) $(TESSDATA_DIR)

annotate:
	$(info Annotating final output...)
ifeq ($(CROP_PEOPLE),1)
	python3 annotate.py $(OUT_DIR)/$(JOB_ID)/input $(OUT_DIR)/$(JOB_ID)/out/annotated $(OUT_DIR)/$(JOB_ID)/out/text $(OUT_DIR)/$(JOB_ID)/out/chars $(OUT_DIR)/$(JOB_ID)/out/aggregate
else
	python3 annotate.py $(OUT_DIR)/$(JOB_ID)/input $(OUT_DIR)/$(JOB_ID)/out/annotated $(OUT_DIR)/$(JOB_ID)/out/text $(OUT_DIR)/$(JOB_ID)/out/chars $(OUT_DIR)/$(JOB_ID)/out/bib
endif

measure_accuracy:
	$(error TODO)
# Check out/job_id/gt.json file (made from prepare)
# if not exist, error
# if exist then run measure_accuracy with file and spit out out/accuracy.json
