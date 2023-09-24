# How to run and reproduce algorithm container

## Clone this repository and install packages
`pip install .`

## Download latest.pth
Download latest.pth from this [Google Drive](https://drive.google.com/drive/folders/1gPzgCTRPB-22JwvGlQBSwVQLGS9DgxQD?usp=sharing)

Put the latest.pth to mmdetection/work_dirs/

## run export.sh
`sudo bash export.sh`

The naming convention for the output files is as follows.
```
current_date_time=$(date '+%Y%m%d_%H%M%S')
output_file="surgtoolloc_det_${current_date_time}.tar.gz"
```
