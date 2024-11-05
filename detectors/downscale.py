
from moviepy.editor import *
import sys


def downscale(input_path, output_path):
    clip = VideoFileClip(input_path).set_fps(10)
    clip.write_videofile(output_path, codec="libx264")

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    downscale(input_path, output_path)


