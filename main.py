import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sys

if len(sys.argv) != 2:
    print("Usage: python3 main.py <image_path>")
    sys.exit(1)

img_file = sys.argv[1]

img = Image.open(img_file)

font_size = 12
font = ImageFont.truetype("CascadiaMono.ttf", font_size)

UNICODE_RANGES = [
    (32, 127),      # Basic ASCII
    (160, 255),     # Latin-1 Supplement
    (0x0100, 0x017F), # Latin Extended-A
    (0x2500, 0x257F), # Box Drawing
    (0x2580, 0x259F), # Block Elements
    (0x25A0, 0x25FF), # Geometric Shapes
    (0x2800, 0x28FF), # Braille Patterns
]

if not os.path.exists("char_images"):
    os.makedirs("char_images")
    for start, end in UNICODE_RANGES:
        for char_code in range(start, end + 1):
            try:
                char = chr(char_code)
                if char.isprintable() or char_code in range(0x2500, 0x25FF):
                    img_char = Image.new("L", (font_size, int(font_size * 1.2)), 255)
                    draw = ImageDraw.Draw(img_char)
                    center_x, center_y = int(font_size * 0.5), int(font_size * 0.6)
                    draw.text((center_x, center_y), char, font=font, fill=0, anchor="mm")
                    img_char.save(f"char_images/{char_code}.png")
            except (UnicodeError, OSError):
                pass

terminal_width, terminal_height = os.get_terminal_size()

img_width, img_height = img.size

max_chars_width = terminal_width - 2
max_chars_height = terminal_height - 2

target_width = max_chars_width * font_size
target_height = max_chars_height * font_size

img_aspect = 1.2 * img_width / img_height
target_aspect = target_width / target_height

if img_aspect > target_aspect:
    new_width = target_width
    new_height = int(target_width / img_aspect)
else:
    new_height = target_height
    new_width = int(target_height * img_aspect)

img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

final_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))

x_offset = (target_width - new_width) // 2
y_offset = (target_height - new_height) // 2

final_img.paste(img_resized, (x_offset, y_offset))

color_img = final_img
gray_img = final_img.convert("L")

final_width, final_height = final_img.size

def rgb_to_ansi(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

def reset_color():
    return "\033[0m"

def rgb_to_ansi_fg_bg(fg_r, fg_g, fg_b, bg_r, bg_g, bg_b):
    return f"\033[38;2;{fg_r};{fg_g};{fg_b}m\033[48;2;{bg_r};{bg_g};{bg_b}m"

def get_text_and_bg_colors(color_chunk, char_img):
    color_pixels = list(color_chunk.getdata())
    char_pixels = list(char_img.getdata())
    
    text_pixels = []
    bg_pixels = []
    
    threshold = 127
    
    for color_pixel, char_pixel in zip(color_pixels, char_pixels):
        if char_pixel < threshold:
            text_pixels.append(color_pixel)
        else:
            bg_pixels.append(color_pixel)
    
    if text_pixels:
        text_r = sum(p[0] for p in text_pixels) // len(text_pixels)
        text_g = sum(p[1] for p in text_pixels) // len(text_pixels)
        text_b = sum(p[2] for p in text_pixels) // len(text_pixels)
    else:
        text_r = text_g = text_b = 255
    
    if bg_pixels:
        bg_r = sum(p[0] for p in bg_pixels) // len(bg_pixels)
        bg_g = sum(p[1] for p in bg_pixels) // len(bg_pixels)
        bg_b = sum(p[2] for p in bg_pixels) // len(bg_pixels)
    else:
        bg_r = bg_g = bg_b = 0
    
    return (text_r, text_g, text_b), (bg_r, bg_g, bg_b)

chunk_height = int(font_size * 1.2)
chars_height = final_height // chunk_height
chars_width = final_width // font_size

chunks = [[None for _ in range(chars_width)] for _ in range(chars_height)]
color_chunks = [[None for _ in range(chars_width)] for _ in range(chars_height)]

for row in range(chars_height):
    for col in range(chars_width):
        y = row * chunk_height
        x = col * font_size
        
        y_end = min(y + chunk_height, final_height)
        x_end = min(x + font_size, final_width)
        
        if y < final_height and x < final_width:
            gray_chunk = gray_img.crop((x, y, x_end, y_end))
            color_chunk = color_img.crop((x, y, x_end, y_end))
            
            if gray_chunk.size != (font_size, chunk_height):
                gray_chunk = gray_chunk.resize((font_size, chunk_height), Image.Resampling.LANCZOS)
                color_chunk = color_chunk.resize((font_size, chunk_height), Image.Resampling.LANCZOS)
                
            chunks[row][col] = gray_chunk
            color_chunks[row][col] = color_chunk

char_images = {}
for start, end in UNICODE_RANGES:
    for char_code in range(start, end + 1):
        char_file = f"char_images/{char_code}.png"
        if os.path.exists(char_file):
            char_img = Image.open(char_file)
            char_images[char_code] = char_img

def initialize_fast_lookup():
    patterns = []
    char_codes = []
    char_lookup = {}
    
    for char_code, char_img in char_images.items():
        arr = np.array(char_img)
        binary = (arr < 127).astype(np.uint8).flatten()
        
        patterns.append(binary)
        char_codes.append(char_code)
        char_lookup[char_code] = char_img
    
    return np.array(patterns), char_codes, char_lookup

patterns, char_codes, char_lookup = initialize_fast_lookup()
chunk_cache = {}

def find_closest_char_fast(chunk):
    chunk_bytes = chunk.tobytes()
    chunk_hash = hashlib.md5(chunk_bytes).hexdigest()[:16]
    
    if chunk_hash in chunk_cache:
        return chunk_cache[chunk_hash]
    
    chunk_arr = np.array(chunk)
    threshold = np.median(chunk_arr)
    chunk_binary = (chunk_arr < threshold).astype(np.uint8).flatten()
    
    distances = np.sum(patterns != chunk_binary, axis=1)
    best_idx = np.argmin(distances)
    
    char_code = char_codes[best_idx]
    result = (chr(char_code), char_lookup[char_code])
    
    chunk_cache[chunk_hash] = result
    return result

result_text = []
for row_idx, row in enumerate(chunks):
    line = ""
    for col_idx, chunk in enumerate(row):
        if chunk is not None:
            closest_char, char_img = find_closest_char_fast(chunk)
            
            color_chunk = color_chunks[row_idx][col_idx]
            text_color, bg_color = get_text_and_bg_colors(color_chunk, char_img)
            
            color_code = rgb_to_ansi_fg_bg(*text_color, *bg_color)
            line += color_code + closest_char + reset_color()
        else:
            line += " "
    result_text.append(line)

for line in result_text:
    print(line)







    


