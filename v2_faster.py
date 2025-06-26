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
from itertools import combinations

img = Image.open("img.png")

x, y = os.get_terminal_size()

def v2_faster(img, x, y):
    start = time.time()
    aspect = img.width / img.height
    term_aspect = x / (y*2)

    print(aspect, term_aspect)

    if aspect < term_aspect:
        img = img.resize((int(y*aspect*2), y*2))
    if aspect > term_aspect:
        img = img.resize((int(x*2), int(x/aspect)*2))
    
    resize = time.time()

    print(x, y)
    print(img.width, img.height)

    img_gray = img.convert("L")
    img = img.convert("RGB")

    convert = time.time()

    response = ""

    pixel_time = 0
    transform_time = 0

    for i in range(img.height//2):
        for j in range(img.width):
            fg = img.getpixel((j, i*2))
            bg = img.getpixel((j, i*2+1))
            # ▀ Upper half
            char = "\u2580"
            response += f"\033[38;2;{fg[0]};{fg[1]};{fg[2]}m\033[48;2;{bg[0]};{bg[1]};{bg[2]}m{char}"

        response += "\033[0m\n"

    end = time.time()

    print(f"Time taken:")
    if resize - start > 0:
        print(f"Resize: {resize - start} seconds ({1/(resize - start)} fps)")
    else:
        print("Resize: 0 seconds (inf fps)")
    if convert - resize > 0:
        print(f"Convert: {convert - resize} seconds ({1/(convert - resize)} fps)")
    else:
        print("Convert: 0 seconds (inf fps)")
    if end - convert > 0:
        print(f"Process: {end - convert} seconds ({1/(end - convert)} fps)")
    else:
        print("Process: 0 seconds (inf fps)")
    if pixel_time > 0:
        print(f"Pixel: {pixel_time} seconds ({1/pixel_time} fps)")
    else:
        print("Pixel: 0 seconds (inf fps)")
    if transform_time > 0:
        print(f"Transform: {transform_time} seconds ({1/transform_time} fps)")
    else:
        print("Transform: 0 seconds (inf fps)")
    if end - start > 0:
        print(f"Total: {end - start} seconds ({1/(end - start)} fps)")
    else:
        print("Total: 0 seconds (inf fps)")

    return response

print(v2_faster(img, x, y))