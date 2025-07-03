# ⚠️ **PHOTOSENSITIVITY/EPILEPSY WARNING** ⚠️ 

## The video playback functionality contains rapidly flashing lights and colors that may trigger seizures in people with photosensitive epilepsy or other photosensitivities. Viewer discretion is strongly advised.




# IMG2TUI

Shows an image or video on the terminal using ASCII art with full color support.

## Features

- Display videos and images in terminal
- Support for YouTube videos (direct URL input)
- Full RGB color support with ANSI escape codes
- 3 different quality presets (fastest, faster, medium)

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

```bash
python3 main.py [-h] [-q {fastest,faster,medium}] [-f FPS] [-nc] media
```

### Arguments

- `-h`: Show help message and exit
- `-q {fastest,faster,medium}`: Set quality preset (default: faster)
- `-f FPS`: Set target FPS (default: None)
- `-nc`: Disable clearing the screen between frames, can cause flickering (default: True)

### Examples

#### Display an image in medium quality
```bash
python3 main.py -q medium image.png
```

#### Play a video file
```bash
python3 main.py video.mp4
```

#### Play a YouTube video
```bash
python3 main.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

### Supported formats
- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF, WebP
- **Videos**: MP4, AVI, MOV, MKV, WEBM, FLV, OGG, GIF
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
- [ ] Audio support

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)