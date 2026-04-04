# !/usr/bin/sh

python ./yadisk/yadisk_${1}.py \
    --local ./runs \
    --remote-root "disk:/raw-diplom/runs" \
    --token "XXX" \
    --overwrite-different-md5

