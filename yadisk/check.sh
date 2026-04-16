#!/usr/bin/bash

source "$(dirname "$0")/SECRETS.env"

python ./yadisk/yadisk_check_families.py --remote-root "disk:/raw-diplom/runs/$2" --token "$TOKEN"
