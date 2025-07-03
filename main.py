import v2_fastest
import v2_faster
import v2_medium
import os
from PIL import Image
import sys
import cv2
import time
import yt_dlp
import tempfile
import argparse

def main():
    parser = argparse.ArgumentParser(description='Display images/videos in terminal with ASCII art')
    parser.add_argument('media', help='Path to image/video file or YouTube URL')
    parser.add_argument('-q', '--quality', choices=['fastest', 'faster', 'medium'], default='default')
    parser.add_argument('-f', '--fps', type=int, default=None)
    parser.add_argument('-nc', '--no-clear', action='store_false', default=True)
    
    args = parser.parse_args()

    if args.media.startswith('https://www.youtube.com/watch?v=') or args.media.startswith('https://youtu.be/'):
        path = download_video(args.media)
        video(path, args.quality, args.fps, args.no_clear)

    elif args.media.endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm', '.gif', '.ogg')):
        video(args.media, args.quality, args.fps, args.no_clear)

    elif args.media.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        img = Image.open(args.media)
        x, y = os.get_terminal_size()

        if args.quality == 'fastest':
            print(v2_fastest.v2_fastest(img, x, y))
        elif args.quality == 'faster':
            print(v2_faster.v2_faster(img, x, y))
        elif args.quality == 'medium' or args.quality == 'default':
            print(v2_medium.v2_medium(img, x, y))

def video(path, quality, target_fps=None, clear=True):
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = fps if target_fps is None else target_fps
    delay = 1/fps

    sys.stdout.write('\033[?25l\033[2J')
    sys.stdout.flush()

    try:
        for _ in range(frames):
            start = time.time()
            x, y = os.get_terminal_size()

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)

            if quality == 'fastest':
                frame = v2_fastest.v2_fastest(img, x, y)
            elif quality == 'faster' or quality == 'default':
                frame = v2_faster.v2_faster(img, x, y)
            elif quality == 'medium':
                frame = v2_medium.v2_medium(img, x, y)

            if clear:
                sys.stdout.write('\033[H' + frame)
            else:
                sys.stdout.write(frame)
            
            sys.stdout.flush()

            end = time.time()
            time.sleep(max(0, delay - (end - start)))

    except KeyboardInterrupt:
        print("\nPlayback stopped by user")
    finally:
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()
        cap.release()

def download_video(url):
    ydl_opts = {
        'outtmpl': tempfile.gettempdir() + '/%(title)s.%(ext)s',
        'format': 'best[height<=720]'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

if __name__ == "__main__":
    main()