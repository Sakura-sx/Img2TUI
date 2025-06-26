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
import queue
import threading
from collections import defaultdict

MAX_FRAME_BUFFER = 8
MAX_PROCESSED_BUFFER = 16

QUALITY_PRESETS = {
    'fastest': {
        'name': 'Fastest (Background only)',
        'ranges': [],
        'use_background_only': True,
        'char_resolution_scale': 1.0,
        'color_resolution_scale': 0.25,
        'direct_sampling': True,
        'description': 'Colored blocks only - maximum performance'
    },
    'fast': {
        'name': 'Fast (Block Elements)',
        'ranges': [(32, 32), (0x2580, 0x259F)],  # Block Elements
        'use_background_only': False,
        'char_resolution_scale': 1.0,
        'color_resolution_scale': 1.0,
        'direct_sampling': False,
        'description': 'Block Elements only'
    },
    'normal': {
        'name': 'Normal (Block Elements + Geometric Shapes)',
        'ranges': [(32, 32), (0x2580, 0x259F), (0x25A0, 0x25FF)],  # Block Elements, Geometric Shapes
        'use_background_only': False,
        'char_resolution_scale': 1.0,
        'color_resolution_scale': 1.0,
        'direct_sampling': False,
        'description': 'Block Elements and Geometric Shapes'
    },
    'detailed': {
        'name': 'Detailed (Block Elements + Geometric Shapes + Box Drawing)',
        'ranges': [(32, 127), (160, 255), (0x2500, 0x257F), (0x2580, 0x259F)], # Block Elements, Geometric Shapes, Box Drawing
        'use_background_only': False,
        'char_resolution_scale': 1.0,
        'color_resolution_scale': 1.0,
        'direct_sampling': False,
        'description': 'Block Elements, Geometric Shapes, and Box Drawing'
    },
    'slowest': {
        'name': 'Slowest (All Unicode)',
        'ranges': [(32, 127), (160, 255), (0x0100, 0x017F), (0x2500, 0x257F), 
                   (0x2580, 0x259F), (0x25A0, 0x25FF), (0x2800, 0x28FF)],
        'use_background_only': False,
        'char_resolution_scale': 1.0,
        'color_resolution_scale': 1.0,
        'direct_sampling': False,
        'description': 'All Unicode ranges - highest quality'
    }
}

PRESET_ORDER = ['fastest', 'fast', 'normal', 'detailed', 'slowest']

TERMINAL_WIDTH, TERMINAL_HEIGHT = os.get_terminal_size()
FONT_SIZE = 12
MAX_CHARS_WIDTH = TERMINAL_WIDTH - 2
MAX_CHARS_HEIGHT = TERMINAL_HEIGHT - 3
TARGET_WIDTH = MAX_CHARS_WIDTH * FONT_SIZE
TARGET_HEIGHT = MAX_CHARS_HEIGHT * FONT_SIZE
CHUNK_HEIGHT = int(FONT_SIZE * 1.2)

RESET_COLOR = "\033[0m"
CLEAR_SCREEN = "\033[2J\033[H"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Display images/videos in terminal with ASCII art')
    parser.add_argument('media', help='Path to image/video file or YouTube URL')
    parser.add_argument('-q', '--quality', choices=QUALITY_PRESETS.keys(), default='normal',
                       help='Quality preset (default: normal)')
    parser.add_argument('--list-presets', action='store_true', 
                       help='List available quality presets')
    parser.add_argument('--no-auto-adjust', action='store_true',
                       help='Disable automatic quality adjustment for videos')
    parser.add_argument('--threads', type=int, default=8,
                       help='Number of frame processing threads (default: 8)')
    parser.add_argument('--precompute', action='store_true',
                       help='Precompute all frames before playback for smoother performance')
    parser.add_argument('--max-memory-frames', type=int, default=1000,
                       help='Maximum frames to keep in memory during precomputation (default: 1000)')
    return parser.parse_args()

def list_presets():
    print("Available quality presets:")
    for preset_name in PRESET_ORDER:
        preset = QUALITY_PRESETS[preset_name]
        char_scale = preset.get('char_resolution_scale', 1.0)
        color_scale = preset.get('color_resolution_scale', 1.0)
        scale_info = f" (Char: {char_scale:.2f}x, Color: {color_scale:.2f}x)" if char_scale != 1.0 or color_scale != 1.0 else ""
        print(f"  {preset_name:8} - {preset['description']}{scale_info}")

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
    char_scale = preset.get('char_resolution_scale', 1.0)
    color_scale = preset.get('color_resolution_scale', 1.0)
    if char_scale != 1.0 or color_scale != 1.0:
        print(f"Character resolution: {char_scale:.2f}x, Color resolution: {color_scale:.2f}x")
    
    if preset['use_background_only']:
        return {}, np.array([]), [], {}
    
    font = ImageFont.truetype("CascadiaMono.ttf", FONT_SIZE)
    
    char_font_size = int(FONT_SIZE * char_scale)
    char_chunk_height = int(char_font_size * 1.2)
    
    if not os.path.exists("char_images"):
        os.makedirs("char_images")
    
    for start, end in preset['ranges']:
        for char_code in range(start, end + 1):
            char_file = f"char_images/{char_code}_{char_scale:.2f}.png"
            if not os.path.exists(char_file):
                try:
                    char = chr(char_code)
                    if char.isprintable() or char_code in range(0x2500, 0x25FF):
                        img_char = Image.new("L", (char_font_size, char_chunk_height), 255)
                        draw = ImageDraw.Draw(img_char)
                        center_x, center_y = int(char_font_size * 0.5), int(char_font_size * 0.6)
                        draw.text((center_x, center_y), char, font=font, fill=0, anchor="mm")
                        if char_scale != 1.0:
                            img_char = img_char.resize((char_font_size, char_chunk_height), Image.Resampling.LANCZOS)
                        img_char.save(char_file)
                except (UnicodeError, OSError):
                    pass
    
    char_images = {}
    char_arrays = {}
    for start, end in preset['ranges']:
        for char_code in range(start, end + 1):
            char_file = f"char_images/{char_code}_{char_scale:.2f}.png"
            if os.path.exists(char_file):
                char_img = Image.open(char_file)
                char_images[char_code] = char_img
                char_arrays[char_code] = np.array(char_img)
    
    patterns = []
    char_codes = []
    char_lookup = {}
    
    for char_code, char_arr in char_arrays.items():
        binary = (char_arr < 127).astype(np.uint8).flatten()
        
        patterns.append(binary)
        char_codes.append(char_code)
        char_lookup[char_code] = char_images[char_code]
    
    return char_images, np.array(patterns), char_codes, char_lookup

def ensure_rgb_image(img_array):
    if len(img_array.shape) == 2:  # Grayscale
        return cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        return img_array
    else:
        raise ValueError(f"Unsupported image format with shape: {img_array.shape}")

def downsample_image(img_array, scale):
    if scale >= 1.0:
        return img_array
    
    h, w = img_array.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    
    if len(img_array.shape) == 3:
        return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

def upsample_to_match(small_array, target_shape):
    if len(small_array.shape) == 3:
        return cv2.resize(small_array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        return cv2.resize(small_array, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

def get_direct_sample_color(color_array):
    if color_array.size == 0:
        return (0, 0, 0)
    
    if len(color_array.shape) == 2:
        center_val = color_array[color_array.shape[0]//2, color_array.shape[1]//2]
        return (int(center_val), int(center_val), int(center_val))
    elif len(color_array.shape) == 3 and color_array.shape[2] == 3:
        h, w = color_array.shape[:2]
        center_y, center_x = h // 2, w // 2
        color = color_array[center_y, center_x]
        return (int(color[0]), int(color[1]), int(color[2]))
    else:
        return (0, 0, 0)

def get_average_color_vectorized(color_array):
    if color_array.size == 0:
        return (0, 0, 0)
    
    if len(color_array.shape) == 2:
        avg_val = int(np.mean(color_array))
        return (avg_val, avg_val, avg_val)
    elif len(color_array.shape) == 3 and color_array.shape[2] == 3:
        avg_color = np.mean(color_array.reshape(-1, 3), axis=0).astype(np.uint8)
        return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
    else:
        return (0, 0, 0)

def get_text_and_bg_colors_vectorized(color_array, char_array):
    threshold = 127
    
    if len(color_array.shape) == 2:
        color_array = np.stack([color_array, color_array, color_array], axis=2)
    
    color_h, color_w = color_array.shape[:2]
    char_h, char_w = char_array.shape[:2]
    
    if color_h != char_h or color_w != char_w:
        char_array_resized = cv2.resize(char_array, (color_w, color_h), interpolation=cv2.INTER_NEAREST)
    else:
        char_array_resized = char_array
    
    color_flat = color_array.reshape(-1, 3)
    char_flat = char_array_resized.flatten()
    
    text_mask = char_flat < threshold
    bg_mask = ~text_mask
    
    if np.any(text_mask):
        text_color = np.mean(color_flat[text_mask], axis=0).astype(np.uint8)
    else:
        text_color = np.array([255, 255, 255], dtype=np.uint8)
    
    if np.any(bg_mask):
        bg_color = np.mean(color_flat[bg_mask], axis=0).astype(np.uint8)
    else:
        bg_color = np.array([0, 0, 0], dtype=np.uint8)
    
    return (int(text_color[0]), int(text_color[1]), int(text_color[2])), (int(bg_color[0]), int(bg_color[1]), int(bg_color[2]))

def find_closest_char_vectorized(chunk_array, patterns, char_codes, char_lookup, chunk_cache):
    chunk_bytes = chunk_array.tobytes()
    chunk_hash = hashlib.md5(chunk_bytes).hexdigest()[:12]
    
    if chunk_hash in chunk_cache:
        return chunk_cache[chunk_hash]
    
    threshold = np.median(chunk_array)
    chunk_binary = (chunk_array < threshold).astype(np.uint8).flatten()
    
    distances = np.sum(patterns != chunk_binary, axis=1)
    best_idx = np.argmin(distances)
    
    char_code = char_codes[best_idx]
    result = (chr(char_code), char_lookup[char_code])
    
    chunk_cache[chunk_hash] = result
    return result

def process_chunk_row_background_only_optimized(row_data, direct_sampling=False):
    row_idx, color_arrays_row = row_data
    line_parts = []
    
    for color_array in color_arrays_row:
        if color_array is not None:
            if direct_sampling:
                bg_color = get_direct_sample_color(color_array)
            else:
                bg_color = get_average_color_vectorized(color_array)
            line_parts.append(f"\033[48;2;{bg_color[0]};{bg_color[1]};{bg_color[2]}m \033[0m")
        else:
            line_parts.append(" ")
    
    return row_idx, ''.join(line_parts)

def process_chunk_row_with_chars_optimized(row_data, patterns, char_codes, char_lookup, chunk_cache):
    row_idx, gray_arrays_row, color_arrays_row = row_data
    line_parts = []
    
    for gray_array, color_array in zip(gray_arrays_row, color_arrays_row):
        if gray_array is not None and color_array is not None:
            closest_char, char_img = find_closest_char_vectorized(gray_array, patterns, char_codes, char_lookup, chunk_cache)
            
            char_array = np.array(char_img)
            text_color, bg_color = get_text_and_bg_colors_vectorized(color_array, char_array)
            
            line_parts.append(f"\033[38;2;{text_color[0]};{text_color[1]};{text_color[2]}m\033[48;2;{bg_color[0]};{bg_color[1]};{bg_color[2]}m{closest_char}\033[0m")
        else:
            line_parts.append(" ")
    
    return row_idx, ''.join(line_parts)

def process_frame_optimized(img, quality_preset, patterns, char_codes, char_lookup, chunk_cache):
    preset = QUALITY_PRESETS[quality_preset]
    char_scale = preset.get('char_resolution_scale', 1.0)
    color_scale = preset.get('color_resolution_scale', 1.0)
    
    img_width, img_height = img.size
    img_aspect = 1.2 * img_width / img_height
    target_aspect = TARGET_WIDTH / TARGET_HEIGHT

    if img_aspect > target_aspect:
        new_width = TARGET_WIDTH
        new_height = int(TARGET_WIDTH / img_aspect)
    else:
        new_height = TARGET_HEIGHT
        new_width = int(TARGET_HEIGHT * img_aspect)

    img_array = np.array(img)
    img_array = ensure_rgb_image(img_array)
    
    img_resized = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    final_img = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    x_offset = (TARGET_WIDTH - new_width) // 2
    y_offset = (TARGET_HEIGHT - new_height) // 2
    final_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_resized

    char_font_size = int(FONT_SIZE * char_scale)
    char_chunk_height = int(char_font_size * 1.2)
    color_font_size = int(FONT_SIZE * color_scale)
    color_chunk_height = int(color_font_size * 1.2)

    if preset['use_background_only']:
        color_img_downsampled = downsample_image(final_img, color_scale)
        
        color_arrays_rows = []
        for row in range(MAX_CHARS_HEIGHT):
            color_arrays_row = []
            for col in range(MAX_CHARS_WIDTH):
                y = int(row * color_chunk_height)
                x = int(col * color_font_size)
                y_end = min(y + color_chunk_height, color_img_downsampled.shape[0])
                x_end = min(x + color_font_size, color_img_downsampled.shape[1])
                
                if y < color_img_downsampled.shape[0] and x < color_img_downsampled.shape[1]:
                    color_chunk = color_img_downsampled[y:y_end, x:x_end]
                    color_arrays_row.append(color_chunk)
                else:
                    color_arrays_row.append(None)
            color_arrays_rows.append(color_arrays_row)
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            row_data = [(i, color_arrays_rows[i]) for i in range(len(color_arrays_rows))]
            process_func = lambda rd: process_chunk_row_background_only_optimized(rd, preset['direct_sampling'])
            results = list(executor.map(process_func, row_data))
    
    else:
        gray_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2GRAY)
        gray_img_downsampled = downsample_image(gray_img, char_scale)
        color_img_downsampled = downsample_image(final_img, color_scale)
        
        gray_arrays_rows = []
        color_arrays_rows = []
        
        for row in range(MAX_CHARS_HEIGHT):
            gray_arrays_row = []
            color_arrays_row = []
            for col in range(MAX_CHARS_WIDTH):
                gray_y = int(row * char_chunk_height)
                gray_x = int(col * char_font_size)
                gray_y_end = min(gray_y + char_chunk_height, gray_img_downsampled.shape[0])
                gray_x_end = min(gray_x + char_font_size, gray_img_downsampled.shape[1])
                
                color_y = int(row * color_chunk_height)
                color_x = int(col * color_font_size)
                color_y_end = min(color_y + color_chunk_height, color_img_downsampled.shape[0])
                color_x_end = min(color_x + color_font_size, color_img_downsampled.shape[1])
                
                if (gray_y < gray_img_downsampled.shape[0] and gray_x < gray_img_downsampled.shape[1] and
                    color_y < color_img_downsampled.shape[0] and color_x < color_img_downsampled.shape[1]):
                    
                    gray_chunk = gray_img_downsampled[gray_y:gray_y_end, gray_x:gray_x_end]
                    color_chunk = color_img_downsampled[color_y:color_y_end, color_x:color_x_end]
                    
                    if gray_chunk.shape != (char_chunk_height, char_font_size):
                        gray_chunk = cv2.resize(gray_chunk, (char_font_size, char_chunk_height), interpolation=cv2.INTER_LANCZOS4)
                    
                    if color_chunk.shape[:2] != (color_chunk_height, color_font_size):
                        color_chunk = cv2.resize(color_chunk, (color_font_size, color_chunk_height), interpolation=cv2.INTER_LANCZOS4)
                    
                    gray_arrays_row.append(gray_chunk)
                    color_arrays_row.append(color_chunk)
                else:
                    gray_arrays_row.append(None)
                    color_arrays_row.append(None)
                    
            gray_arrays_rows.append(gray_arrays_row)
            color_arrays_rows.append(color_arrays_row)
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            row_data = [(i, gray_arrays_rows[i], color_arrays_rows[i]) for i in range(len(gray_arrays_rows))]
            process_func = lambda rd: process_chunk_row_with_chars_optimized(rd, patterns, char_codes, char_lookup, chunk_cache)
            results = list(executor.map(process_func, row_data))
    
    results.sort(key=lambda x: x[0])
    return [line for _, line in results]

def get_quality_adjustment(current_fps, target_fps, current_preset):
    current_idx = PRESET_ORDER.index(current_preset)
    
    if current_fps < target_fps * 0.7 and current_idx > 0:
        return PRESET_ORDER[current_idx - 1]
    
    elif current_fps > target_fps * 1.2 and current_idx < len(PRESET_ORDER) - 1:
        return PRESET_ORDER[current_idx + 1]
    
    return current_preset

def precompute_frame_worker(frame_data, quality_preset, patterns, char_codes, char_lookup):
    frame_number, frame = frame_data
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    
    chunk_cache = {}
    result_text = process_frame_optimized(img, quality_preset, patterns, char_codes, char_lookup, chunk_cache)
    return frame_number, result_text

def precompute_video_frames(cap, quality_preset, patterns, char_codes, char_lookup, max_threads=4, max_memory_frames=1000):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Precomputing {total_frames} frames...")
    print(f"Using {max_threads} threads for precomputation")
    
    precomputed_frames = {}
    temp_files = {}
    frames_in_memory = 0
    
    frame_data = []
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_data.append((frame_number, frame))
        frame_number += 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        batch_size = max_threads * 4
        
        for i in range(0, len(frame_data), batch_size):
            batch = frame_data[i:i + batch_size]
            
            futures = []
            for frame_info in batch:
                future = executor.submit(precompute_frame_worker, frame_info, quality_preset, patterns, char_codes, char_lookup)
                futures.append(future)
            
            for future in futures:
                frame_num, result_text = future.result()
                
                if frames_in_memory < max_memory_frames:
                    precomputed_frames[frame_num] = result_text
                    frames_in_memory += 1
                else:
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                    temp_file.write('\n'.join(result_text))
                    temp_file.close()
                    temp_files[frame_num] = temp_file.name
                
                progress = ((frame_num + 1) / total_frames) * 100
                elapsed = time.time() - start_time
                
                if frame_num > 0:
                    eta = (elapsed / (frame_num + 1)) * (total_frames - frame_num - 1)
                    eta_min = int(eta // 60)
                    eta_sec = int(eta % 60)
                    print(f"\rProgress: {progress:.1f}% ({frame_num + 1}/{total_frames}) - ETA: {eta_min}:{eta_sec:02d}", end="", flush=True)
    
    print(f"\nPrecomputation completed in {time.time() - start_time:.1f}s")
    print(f"Frames in memory: {frames_in_memory}, Frames on disk: {len(temp_files)}")
    
    return precomputed_frames, temp_files, total_frames, fps

def get_precomputed_frame(frame_number, precomputed_frames, temp_files):
    if frame_number in precomputed_frames:
        return precomputed_frames[frame_number]
    elif frame_number in temp_files:
        with open(temp_files[frame_number], 'r') as f:
            content = f.read().strip()
            return content.split('\n')
    else:
        return None

def cleanup_temp_files(temp_files):
    for temp_file in temp_files.values():
        try:
            os.unlink(temp_file)
        except OSError:
            pass

def frame_reader_thread(cap, frame_queue, stop_event):
    frame_number = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            frame_queue.put((frame_number, frame), timeout=0.1)
            frame_number += 1
        except queue.Full:
            continue
    
    frame_queue.put(None)

def frame_processor_thread(frame_queue, processed_queue, quality_data, stop_event):
    current_quality, patterns, char_codes, char_lookup = quality_data
    local_chunk_cache = {}
    
    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=0.1)
            if item is None:
                processed_queue.put(None)
                break
            
            frame_number, frame = item
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            result_text = process_frame_optimized(img, current_quality, patterns, char_codes, char_lookup, local_chunk_cache)
            
            processed_queue.put((frame_number, result_text))
            frame_queue.task_done()
            
        except queue.Empty:
            continue

class ConcurrentFrameProcessor:
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        self.frame_queue = queue.Queue(maxsize=MAX_FRAME_BUFFER)
        self.processed_queue = queue.Queue(maxsize=MAX_PROCESSED_BUFFER)
        self.stop_event = threading.Event()
        self.reader_thread = None
        self.processor_threads = []
        self.frame_buffer = {}
        self.next_display_frame = 0
        
    def start(self, cap, quality_data):
        self.stop_event.clear()
        
        self.reader_thread = threading.Thread(
            target=frame_reader_thread,
            args=(cap, self.frame_queue, self.stop_event)
        )
        self.reader_thread.start()
        
        for i in range(self.num_threads):
            thread = threading.Thread(
                target=frame_processor_thread,
                args=(self.frame_queue, self.processed_queue, quality_data, self.stop_event)
            )
            thread.start()
            self.processor_threads.append(thread)
    
    def get_next_frame(self, timeout=1.0):
        while True:
            if self.next_display_frame in self.frame_buffer:
                result = self.frame_buffer.pop(self.next_display_frame)
                self.next_display_frame += 1
                return result
            
            try:
                item = self.processed_queue.get(timeout=timeout)
                if item is None:
                    return None
                
                frame_number, result_text = item
                if frame_number == self.next_display_frame:
                    self.next_display_frame += 1
                    return result_text
                else:
                    self.frame_buffer[frame_number] = result_text
                    
            except queue.Empty:
                return "TIMEOUT"
    
    def update_quality(self, quality_data):
        self.stop()
        self.frame_buffer.clear()
        self.next_display_frame = 0
        
    def stop(self):
        self.stop_event.set()
        
        if self.reader_thread:
            self.reader_thread.join()
        
        for thread in self.processor_threads:
            thread.join()
        
        self.processor_threads.clear()
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.processed_queue.empty():
            try:
                self.processed_queue.get_nowait()
            except queue.Empty:
                break

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
        
        if is_video:
            cap = cv2.VideoCapture(media_file)
            
            if not cap.isOpened():
                print(f"Error: Could not open video file {media_file}")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video: {fps:.1f} FPS, {total_frames} frames")
            
            if args.precompute:
                precomputed_frames, temp_files, total_frames, fps = precompute_video_frames(
                    cap, current_quality, patterns, char_codes, char_lookup, 
                    args.threads, args.max_memory_frames
                )
                
                frame_count = 0
                start_time = time.time()
                
                print("Starting precomputed playback. Press Ctrl+C to stop")
                
                try:
                    while frame_count < total_frames:
                        frame_start = time.time()
                        
                        result_text = get_precomputed_frame(frame_count, precomputed_frames, temp_files)
                        if result_text is None:
                            break
                        
                        print(CLEAR_SCREEN, end="")
                        
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        elapsed_min = int(elapsed_time // 60)
                        elapsed_sec = int(elapsed_time % 60)
                        progress = (frame_count / total_frames) * 100
                        
                        preset_name = QUALITY_PRESETS[current_quality]['name']
                        print(f"\033[33m{preset_name} (Precomputed) | Progress: {progress:.1f}% | Time: {elapsed_min}:{elapsed_sec:02d}\033[0m")
                        
                        if isinstance(result_text, list):
                            print('\n'.join(result_text))
                        else:
                            print(result_text)
                        
                        frame_count += 1
                        
                        frame_end = time.time()
                        processing_time = frame_end - frame_start
                        
                        remaining_time = frame_delay - processing_time
                        if remaining_time > 0:
                            time.sleep(remaining_time)
                            
                except KeyboardInterrupt:
                    print("\nPlayback stopped by user")
                finally:
                    cleanup_temp_files(temp_files)
                    cap.release()
            else:
                frame_count = 0
                start_time = time.time()
                last_fps_update = start_time
                last_quality_check = start_time
                current_fps = 0
                
                print(f"Using {args.threads} processing threads")
                if not args.no_auto_adjust:
                    print("Auto quality adjustment enabled")
                print("Press Ctrl+C to stop")
                
                processor = ConcurrentFrameProcessor(args.threads)
                quality_data = (current_quality, patterns, char_codes, char_lookup)
                processor.start(cap, quality_data)
                
                try:
                    while True:
                        frame_start = time.time()
                        
                        result_text = processor.get_next_frame(timeout=2.0)
                        if result_text is None:
                            print("\nVideo finished")
                            break
                        elif result_text == "TIMEOUT":
                            continue
                        
                        print(CLEAR_SCREEN, end="")
                        
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
                                quality_data = (current_quality, patterns, char_codes, char_lookup)
                                processor.stop()
                                processor = ConcurrentFrameProcessor(args.threads)
                                processor.start(cap, quality_data)
                            last_quality_check = current_time
                        
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        elapsed_time = current_time - start_time
                        elapsed_min = int(elapsed_time // 60)
                        elapsed_sec = int(elapsed_time % 60)
                        
                        preset_name = QUALITY_PRESETS[current_quality]['name']
                        print(f"\033[33m{preset_name} | FPS: {current_fps:.1f}/{fps:.1f} | Progress: {progress:.1f}% | Time: {elapsed_min}:{elapsed_sec:02d} | Threads: {args.threads}\033[0m")
                        
                        print('\n'.join(result_text))
                        
                        frame_end = time.time()
                        processing_time = frame_end - frame_start
                        
                        remaining_time = frame_delay - processing_time
                        if remaining_time > 0:
                            time.sleep(remaining_time)
                            
                except KeyboardInterrupt:
                    print("\nPlayback stopped by user")
                finally:
                    processor.stop()
                    cap.release()
        else:
            img = Image.open(media_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            chunk_cache = {}
            result_text = process_frame_optimized(img, current_quality, patterns, char_codes, char_lookup, chunk_cache)
            preset_name = QUALITY_PRESETS[current_quality]['name']
            print(f"Quality: {preset_name}")
            print('\n'.join(result_text))
    
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






    


