import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor

img_file = "img.png"
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
                    img = Image.new("L", (font_size, int(font_size * 1.2)), 255)
                    draw = ImageDraw.Draw(img)
                    center_x, center_y = int(font_size * 0.6), font_size // 2
                    draw.text((center_x, center_y), char, font=font, fill=0, anchor="mm")
                    img.save(f"char_images/{char_code}.png")
            except (UnicodeError, OSError):
                pass

termina_width, termina_height = os.get_terminal_size()
print(termina_width, termina_height)

img_width, img_height = img.size

terminal_resolution = (termina_width * font_size, int(termina_height * font_size * 1.2))

img_aspect = img_width / img_height
terminal_aspect = terminal_resolution[0] / terminal_resolution[1]

if img_aspect > terminal_aspect:
    new_width = terminal_resolution[0]
    new_height = int(new_width / img_aspect)
else:
    new_height = terminal_resolution[1]
    new_width = int(new_height * img_aspect)

img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

final_img = Image.new("RGB", terminal_resolution, (0, 0, 0))

x_offset = (terminal_resolution[0] - new_width) // 2
y_offset = (terminal_resolution[1] - new_height) // 2

final_img.paste(img_resized, (x_offset, y_offset))

img = final_img

color_img = img
gray_img = img.convert("L")

img_width, img_height = gray_img.size

def rgb_to_ansi(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

def reset_color():
    return "\033[0m"

def get_average_color(img_chunk):
    pixels = list(img_chunk.getdata())
    if not pixels:
        return (0, 0, 0)
    
    r_sum = sum(pixel[0] for pixel in pixels)
    g_sum = sum(pixel[1] for pixel in pixels) 
    b_sum = sum(pixel[2] for pixel in pixels)
    pixel_count = len(pixels)
    
    return (r_sum // pixel_count, g_sum // pixel_count, b_sum // pixel_count)

def rgb_to_ansi_fg_bg(fg_r, fg_g, fg_b, bg_r, bg_g, bg_b):
    return f"\033[38;2;{fg_r};{fg_g};{fg_b}m\033[48;2;{bg_r};{bg_g};{bg_b}m"

def get_text_and_bg_colors(color_chunk, char_img):
    color_chunk = color_chunk.resize((font_size, int(font_size * 1.2)), Image.Resampling.LANCZOS)
    
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

chunks = [[None for _ in range(termina_width)] for _ in range(termina_height)]
color_chunks = [[None for _ in range(termina_width)] for _ in range(termina_height)]

for y in range(0, img_height, 14):
    for x in range(0, img_width, 12):
        chunk_row = y // 14
        chunk_col = x // 12
        
        if chunk_row < termina_height and chunk_col < termina_width:
            gray_chunk = gray_img.crop((x, y, x + 12, y + 14))
            color_chunk = color_img.crop((x, y, x + 12, y + 14))
            chunks[chunk_row][chunk_col] = gray_chunk
            color_chunks[chunk_row][chunk_col] = color_chunk

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
    
    chunk_resized = chunk.resize((font_size, int(font_size * 1.2)), Image.Resampling.LANCZOS)
    chunk_arr = np.array(chunk_resized)
    threshold = np.median(chunk_arr)
    chunk_binary = (chunk_arr < threshold).astype(np.uint8).flatten()
    
    distances = np.sum(patterns != chunk_binary, axis=1)
    best_idx = np.argmin(distances)
    
    char_code = char_codes[best_idx]
    result = (chr(char_code), char_lookup[char_code])
    
    chunk_cache[chunk_hash] = result
    return result

def rgb_to_ansi_fg_only(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"

def get_text_color_only(color_chunk, char_img):
    color_chunk = color_chunk.resize((font_size, int(font_size * 1.2)), Image.Resampling.LANCZOS)
    
    color_pixels = list(color_chunk.getdata())
    char_pixels = list(char_img.getdata())
    
    text_pixels = []
    
    threshold = 127
    
    for color_pixel, char_pixel in zip(color_pixels, char_pixels):
        if char_pixel < threshold:
            text_pixels.append(color_pixel)
    
    if text_pixels:
        text_r = sum(p[0] for p in text_pixels) // len(text_pixels)
        text_g = sum(p[1] for p in text_pixels) // len(text_pixels)
        text_b = sum(p[2] for p in text_pixels) // len(text_pixels)
    else:
        text_r = text_g = text_b = 255
    
    return (text_r, text_g, text_b)

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







    


