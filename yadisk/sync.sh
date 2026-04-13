#!/usr/bin/bash

source "$(dirname "$0")/SECRETS.env"

for run in "${@:2}"; do
    python ./yadisk/yadisk_${1}.py \
        --local "./runs/$run" \
        --remote-root "disk:/raw-diplom/runs/$run" \
        --token "$TOKEN"
done

