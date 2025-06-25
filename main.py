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

QUALITY_PRESETS = {
    'fastest': {
        'name': 'Fastest (Background only)',
        'ranges': [],
        'use_background_only': True,
        'description': 'Colored blocks only - maximum performance'
    },
    'fast': {
        'name': 'Fast (Block Elements)',
        'ranges': [(0x2580, 0x259F)],  # Block Elements
        'use_background_only': False,
        'description': 'Block Elements only'

    },
    'normal': {
        'name': 'Normal (Block Elements + Geometric Shapes)',
        'ranges': [(0x2580, 0x259F), (0x25A0, 0x25FF)],  # Block Elements, Geometric Shapes
        'use_background_only': False,
        'description': 'Block Elements and Geometric Shapes'
    },
    'detailed': {
        'name': 'Detailed (Block Elements + Geometric Shapes + Box Drawing)',
        'ranges': [(32, 127), (160, 255), (0x2500, 0x257F), (0x2580, 0x259F)], # Block Elements, Geometric Shapes, Box Drawing
        'use_background_only': False,
        'description': 'Block Elements, Geometric Shapes, and Box Drawing'
    },
    'slowest': {
        'name': 'Slowest (All Unicode)',
        'ranges': [(32, 127), (160, 255), (0x0100, 0x017F), (0x2500, 0x257F), 
                   (0x2580, 0x259F), (0x25A0, 0x25FF), (0x2800, 0x28FF)],
        'use_background_only': False,
        'description': 'All Unicode ranges - highest quality'
    }
}

PRESET_ORDER = ['fastest', 'fast', 'normal', 'detailed', 'slowest']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Display images/videos in terminal with ASCII art')
    parser.add_argument('media', help='Path to image/video file or YouTube URL')
    parser.add_argument('-q', '--quality', choices=QUALITY_PRESETS.keys(), default='normal',
                       help='Quality preset (default: normal)')
    parser.add_argument('--list-presets', action='store_true', 
                       help='List available quality presets')
    parser.add_argument('--no-auto-adjust', action='store_true',
                       help='Disable automatic quality adjustment for videos')
    return parser.parse_args()

def list_presets():
    print("Available quality presets:")
    for preset_name in PRESET_ORDER:
        preset = QUALITY_PRESETS[preset_name]
        print(f"  {preset_name:8} - {preset['description']}")

def is_youtube_url(url):
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return youtube_regex.match(url) is not None

def download_youtube_video(url):
    print("Downloading YouTube video...")
    
    temp_dir = tempfile.mkdtemp()
    
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info.get('title', 'Unknown')
        duration = info.get('duration', 0)
        
        print(f"Title: {video_title}")
        if duration:
            minutes = duration // 60
            seconds = duration % 60
            print(f"Duration: {minutes}:{seconds:02d}")
        
        ydl.download([url])
        
        for file in os.listdir(temp_dir):
            if file.endswith(('.mp4', '.webm', '.mkv', '.flv')):
                return os.path.join(temp_dir, file)
        
        raise Exception("Downloaded file not found")

def initialize_character_set(quality_preset):
    preset = QUALITY_PRESETS[quality_preset]
    
    print(f"Using quality preset: {preset['name']}")
    
    if preset['use_background_only']:
        return {}, np.array([]), [], {}
    
    font_size = 12
    font = ImageFont.truetype("CascadiaMono.ttf", font_size)
    
    if not os.path.exists("char_images"):
        os.makedirs("char_images")
    
    for start, end in preset['ranges']:
        for char_code in range(start, end + 1):
            char_file = f"char_images/{char_code}.png"
            if not os.path.exists(char_file):
                try:
                    char = chr(char_code)
                    if char.isprintable() or char_code in range(0x2500, 0x25FF):
                        img_char = Image.new("L", (font_size, int(font_size * 1.2)), 255)
                        draw = ImageDraw.Draw(img_char)
                        center_x, center_y = int(font_size * 0.5), int(font_size * 0.6)
                        draw.text((center_x, center_y), char, font=font, fill=0, anchor="mm")
                        img_char.save(char_file)
                except (UnicodeError, OSError):
                    pass
    
    char_images = {}
    for start, end in preset['ranges']:
        for char_code in range(start, end + 1):
            char_file = f"char_images/{char_code}.png"
            if os.path.exists(char_file):
                char_img = Image.open(char_file)
                char_images[char_code] = char_img
    
    patterns = []
    char_codes = []
    char_lookup = {}
    
    for char_code, char_img in char_images.items():
        arr = np.array(char_img)
        binary = (arr < 127).astype(np.uint8).flatten()
        
        patterns.append(binary)
        char_codes.append(char_code)
        char_lookup[char_code] = char_img
    
    return char_images, np.array(patterns), char_codes, char_lookup

def rgb_to_ansi_fg_bg(fg_r, fg_g, fg_b, bg_r, bg_g, bg_b):
    return f"\033[38;2;{fg_r};{fg_g};{fg_b}m\033[48;2;{bg_r};{bg_g};{bg_b}m"

def rgb_to_ansi_bg_only(bg_r, bg_g, bg_b):
    return f"\033[48;2;{bg_r};{bg_g};{bg_b}m"

def reset_color():
    return "\033[0m"

def get_average_color(color_chunk):
    color_pixels = list(color_chunk.getdata())
    if not color_pixels:
        return (0, 0, 0)
    
    avg_r = sum(p[0] for p in color_pixels) // len(color_pixels)
    avg_g = sum(p[1] for p in color_pixels) // len(color_pixels)
    avg_b = sum(p[2] for p in color_pixels) // len(color_pixels)
    
    return (avg_r, avg_g, avg_b)

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

def find_closest_char_fast(chunk, patterns, char_codes, char_lookup, chunk_cache):
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

def process_chunk_row_background_only(row_data):
    row_idx, chunks_row, color_chunks_row = row_data
    line = ""
    
    for col_idx, chunk in enumerate(chunks_row):
        if chunk is not None:
            color_chunk = color_chunks_row[col_idx]
            bg_color = get_average_color(color_chunk)
            color_code = rgb_to_ansi_bg_only(*bg_color)
            line += color_code + " " + reset_color()
        else:
            line += " "
    
    return row_idx, line

def process_chunk_row_with_chars(row_data, patterns, char_codes, char_lookup, chunk_cache):
    row_idx, chunks_row, color_chunks_row = row_data
    line = ""
    
    for col_idx, chunk in enumerate(chunks_row):
        if chunk is not None:
            closest_char, char_img = find_closest_char_fast(chunk, patterns, char_codes, char_lookup, chunk_cache)
            
            color_chunk = color_chunks_row[col_idx]
            text_color, bg_color = get_text_and_bg_colors(color_chunk, char_img)
            
            color_code = rgb_to_ansi_fg_bg(*text_color, *bg_color)
            line += color_code + closest_char + reset_color()
        else:
            line += " "
    
    return row_idx, line

def process_frame(img, quality_preset, patterns, char_codes, char_lookup, chunk_cache):
    terminal_width, terminal_height = os.get_terminal_size()
    
    max_chars_width = terminal_width - 2
    max_chars_height = terminal_height - 3
    
    font_size = 12
    target_width = max_chars_width * font_size
    target_height = max_chars_height * font_size
    
    img_width, img_height = img.size
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

    preset = QUALITY_PRESETS[quality_preset]
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        row_data = [(i, chunks[i], color_chunks[i]) for i in range(len(chunks))]
        
        if preset['use_background_only']:
            results = list(executor.map(process_chunk_row_background_only, row_data))
        else:
            process_func = lambda rd: process_chunk_row_with_chars(rd, patterns, char_codes, char_lookup, chunk_cache)
            results = list(executor.map(process_func, row_data))
    
    results.sort(key=lambda x: x[0])
    result_text = [line for _, line in results]

    return result_text

def get_quality_adjustment(current_fps, target_fps, current_preset):
    current_idx = PRESET_ORDER.index(current_preset)
    
    if current_fps < target_fps * 0.7 and current_idx > 0:
        return PRESET_ORDER[current_idx - 1]
    
    elif current_fps > target_fps * 1.2 and current_idx < len(PRESET_ORDER) - 1:
        return PRESET_ORDER[current_idx + 1]
    
    return current_preset

def main():
    args = parse_arguments()
    
    if args.list_presets:
        list_presets()
        return
    
    media_file = args.media
    temp_file = None
    
    try:
        if is_youtube_url(media_file):
            temp_file = download_youtube_video(media_file)
            media_file = temp_file
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', ".ogg"]
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        
        is_video = any(media_file.lower().endswith(ext) for ext in video_extensions)
        is_image = any(media_file.lower().endswith(ext) for ext in image_extensions)
        
        if not is_video and not is_image:
            print(f"Unsupported file format. Supported formats:")
            print(f"Images: {', '.join(image_extensions)}")
            print(f"Videos: {', '.join(video_extensions)}")
            print(f"YouTube URLs: Any valid YouTube video URL")
            return
        
        current_quality = args.quality
        char_images, patterns, char_codes, char_lookup = initialize_character_set(current_quality)
        chunk_cache = {}
        
        if is_video:
            cap = cv2.VideoCapture(media_file)
            
            if not cap.isOpened():
                print(f"Error: Could not open video file {media_file}")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            start_time = time.time()
            last_fps_update = start_time
            last_quality_check = start_time
            current_fps = 0
            
            print(f"Playing video: {fps:.1f} FPS, {total_frames} frames")
            if not args.no_auto_adjust:
                print("Auto quality adjustment enabled")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    frame_start = time.time()
                    
                    ret, frame = cap.read()
                    if not ret:
                        print("\nVideo finished")
                        break
                        
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    
                    result_text = process_frame(img, current_quality, patterns, char_codes, char_lookup, chunk_cache)
                    
                    print("\033[2J\033[H", end="")
                    
                    frame_count += 1
                    current_time = time.time()
                    
                    if current_time - last_fps_update >= 1.0:
                        current_fps = frame_count / (current_time - start_time)
                        last_fps_update = current_time
                    
                    if not args.no_auto_adjust and current_time - last_quality_check >= 3.0:
                        new_quality = get_quality_adjustment(current_fps, fps, current_quality)
                        if new_quality != current_quality:
                            print(f"\nAdjusting quality: {current_quality} -> {new_quality}")
                            current_quality = new_quality
                            char_images, patterns, char_codes, char_lookup = initialize_character_set(current_quality)
                            chunk_cache.clear()
                        last_quality_check = current_time
                    
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    elapsed_time = current_time - start_time
                    elapsed_min = int(elapsed_time // 60)
                    elapsed_sec = int(elapsed_time % 60)
                    
                    preset_name = QUALITY_PRESETS[current_quality]['name']
                    print(f"\033[33m{preset_name} | FPS: {current_fps:.1f}/{fps:.1f} | Progress: {progress:.1f}% | Time: {elapsed_min}:{elapsed_sec:02d}\033[0m")
                    
                    for line in result_text:
                        print(line)
                    
                    frame_end = time.time()
                    processing_time = frame_end - frame_start
                    
                    remaining_time = frame_delay - processing_time
                    if remaining_time > 0:
                        time.sleep(remaining_time)
                        
            except KeyboardInterrupt:
                print("\nPlayback stopped by user")
            finally:
                cap.release()
        else:
            img = Image.open(media_file)
            result_text = process_frame(img, current_quality, patterns, char_codes, char_lookup, chunk_cache)
            preset_name = QUALITY_PRESETS[current_quality]['name']
            print(f"Quality: {preset_name}")
            for line in result_text:
                print(line)
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                temp_dir = os.path.dirname(temp_file)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                print(f"Cleaned up temporary files")
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")

if __name__ == "__main__":
    main()






    


