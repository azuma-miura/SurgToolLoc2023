#!/usr/bin/env bash

sudo bash ./build.sh

# 現在の日時を YYYYMMDD_HHMMSS 形式で取得
current_date_time=$(date '+%Y%m%d_%H%M%S')

# 日時をファイル名に含める
output_file="surgtoolloc_det_${current_date_time}.tar.gz"

# Docker イメージを保存して gzip で圧縮
docker save surgtoolloc_det | gzip -c > "$output_file"
