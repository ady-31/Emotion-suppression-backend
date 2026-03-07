import subprocess
import os

OPENFACE_EXE = r"G:\OpenFace_2.2.0_win_x64\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

def extract_aus(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    command = [
        OPENFACE_EXE,
        "-f", video_path,
        "-out_dir", output_dir,
        "-aus"
    ]

    subprocess.run(command)