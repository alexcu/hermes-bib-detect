#!/usr/bin/env bash

DATE=$(date +%d-%m-%Y)

./eval.sh i 1 1 | tee /home/alex/log/eval_$DATE.log.txt
./eval.sh r 1 1 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh i 1 0 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh r 1 0 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh i 100 1 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh r 100 1 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh i 100 0 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh r 100 0 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh i 500 1 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh r 500 1 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh i 500 0 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh r 500 0 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh i all 1 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh r all 1 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh i all 0 | tee -a /home/alex/log/eval_$DATE.log.txt
./eval.sh r all 0 | tee -a /home/alex/log/eval_$DATE.log.txt
