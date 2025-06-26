import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sys
import cv2
import time
import yt_dlp
import tempfile
import re
import argparse

img = Image.open("img.png")

x, y = os.get_terminal_size()

def v2_fastest(img, x, y):
    aspect = img.width / img.height
    term_aspect = x / (y*2)

    print(aspect, term_aspect)

    if aspect < term_aspect:
        img = img.resize((int(y*aspect*2), y))
    if aspect > term_aspect:
        img = img.resize((int(x*2), int(x/aspect)))

    print(x, y)
    print(img.width, img.height)

    img = img.convert("RGB")

    response = ""

    for i in range(img.height):
        for j in range(img.width):
            response += f"\033[38;2;{img.getpixel((j, i))[0]};{img.getpixel((j, i))[1]};{img.getpixel((j, i))[2]}m█"
        response += "\033[0m\n"

    return response

start = time.time()
print(v2_fastest(img, x, y))
end = time.time()
print(f"Time taken: {end - start} seconds ({1/(end - start)} fps)")
