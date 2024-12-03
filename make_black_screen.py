import argparse
import os
from datetime import datetime
from moviepy import TextClip, ColorClip, CompositeVideoClip


parser = argparse.ArgumentParser()
parser.add_argument(
    "--text", "-t", type=str, required=True, help="Text to shown on the black screen"
)
parser.add_argument(
    "--save_dir", type=str, default="save", help="Folder to put the video in"
)
parser.add_argument(
    "--save_name",
    type=str,
    default=datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S") + ".mp4",
    help="Video filename. It should include the extension of the desired video format, e.g., .mp4",
)
args = parser.parse_args()


def make_black_screen(text, duration, size):
    bg_clip = ColorClip(size=size, color=(0, 0, 0), duration=duration)
    text_clip = TextClip(
        text=text,
        font="C:\\Windows\\Fonts\\timesbd.ttf",
        color="white",
        font_size=32,
        size=size,
        method="label",
        text_align="center",
        duration=duration,
    )
    clip = CompositeVideoClip([bg_clip, text_clip])
    return clip


clip = make_black_screen(args.text, 3, (384, 448))
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
clip.write_videofile(os.path.join(args.save_dir, args.save_name), fps=60)
