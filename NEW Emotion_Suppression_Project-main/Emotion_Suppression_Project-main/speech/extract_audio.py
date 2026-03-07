from moviepy import VideoFileClip
import os

def extract_audio(video_path, output_path):

    video = VideoFileClip(video_path)

    if video.audio is None:
        return False  # No audio present

    video.audio.write_audiofile(output_path)
    return True