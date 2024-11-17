# given a folder data with folders with sounds, create a txt listing the files
ls -r data/data/* >> dataset.txt 

# convert that txt to a json
python3 convert_to_json.py

python3 train.py
#python3 inference.py

