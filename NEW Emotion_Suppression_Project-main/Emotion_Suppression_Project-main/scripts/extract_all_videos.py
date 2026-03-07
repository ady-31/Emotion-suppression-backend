import os
import sys

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing.extract_au_openface import extract_aus
VIDEO_ROOT = r"G:\capstone data\CASME II\CASME2_Compressed video\CASME2_compressed"
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw_csv")

for root, dirs, files in os.walk(VIDEO_ROOT):
    for file in files:
        if file.endswith(".avi"):
            video_path = os.path.join(root, file)
            print("Processing:", video_path)
            extract_aus(video_path, OUTPUT_DIR)

print("All videos processed.")