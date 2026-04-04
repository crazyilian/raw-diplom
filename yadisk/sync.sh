# !/usr/bin/sh

python ./yadisk/yadisk_${1}.py \
    --local ./runs/$2 \
    --remote-root "disk:/raw-diplom/runs/$2" \
    --token "XXX" \
    --overwrite-different-md5

