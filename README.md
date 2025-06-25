# IMG2TUI

Shows an image or video on the terminal using ASCII art with full color support.

## Features

- Display images in terminal over 900 unicode characters
- Play videos in terminal with real-time conversion
- Support for YouTube videos (direct URL input)
- Full RGB color support with ANSI escape codes
- Multithreaded

## Installation

### Windows
Has been tested on Windows 11 PowerShell console.
```bash
git clone https://github.com/Sakura-sx/img2tui.git
cd img2tui
python3 -m pip install -r requirements.txt
python3 main.py <image_path_or_video_path_or_youtube_url>
```

### Linux
Not tested yet.
```bash
git clone https://github.com/Sakura-sx/img2tui.git
cd img2tui
python3 -m pip install -r requirements.txt
python3 main.py <image_path_or_video_path_or_youtube_url>
```

## Usage

### Display an image
```bash
python3 main.py image.png
```

### Play a video file
```bash
python3 main.py video.mp4
```

### Play a YouTube video
```bash
python3 main.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Supported formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **Videos**: MP4, AVI, MOV, MKV, WEBM, FLV, OGG
- **YouTube**: Any public YouTube video URL

## Controls

- **Ctrl+C**: Stop video playback

## Roadmap

- [x] Select image on the arguments
- [x] Requirements.txt
- [x] Video support
- [x] YouTube video support
- [x] A way to stream videos and play them
- [x] Different quality presets
- [ ] Speedup video playback

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)