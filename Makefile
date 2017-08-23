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

# Check for required arguments
$(call check_defined, IN_DIR, input directory)
$(call check_defined, OUT_DIR, output directory)
$(call check_defined, DARKNET_DIR, directory to Darknet)
$(call check_defined, PICKLE_CONFIG_BIB, bib pickle config file)
$(call check_defined, PICKLE_CONFIG_TXT, text pickle config file)
$(call check_defined, TESSERACT_BIN_DIR, directory to Tesseract binary)
$(call check_defined, TESSDATA_DIR, tessdata directory)
$(call check_defined, CROP_PEOPLE, whether to crop people -- should be 0 or 1)
$(info Running hermes...)

# Conditionally configure run based on whether we want to crop people
ifeq ($(CROP_PEOPLE),1)
run: make_out_dir \
     person_detect \
     bib_detect \
     person_aggregate \
     text_detect \
     text_recognise
else
run: make_out_dir \
     bib_detect \
     text_detect \
     text_recognise
endif

make_out_dir:
	$(info Creating output directory at $(OUT_DIR)...)
	mkdir -p $(OUT_DIR)

person_detect:
	$(info Running person detection...)

bib_detect:
	$(info Running bib detection...)

person_aggregate:
	$(info Running aggregation of cropped people...)

text_detect:
	$(info Running text detection...)

text_recognise:
	$(info Running text recognition...)
