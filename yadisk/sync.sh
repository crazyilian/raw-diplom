#!/usr/bin/bash

source $(dirname $0)/SECRETS.env

python ./yadisk/yadisk_${1}.py \
    --local ./runs/$2 \
    --remote-root "disk:/raw-diplom/runs/$2" \
    --token $TOKEN
