# Face Monitor

A real-time face recognition system that monitors a folder for new images and automatically identifies faces.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Monitor a folder for new images:

```bash
python face_monitor.py <folder_path>
```

### Options

- `--output`, `-o`: Output file for results (default: `face_recognition_results.txt`)
- `--interval`, `-i`: Check interval in seconds (default: `1.0`)
- `--device`, `-d`: GPU device ID, `-1` for CPU (default: `1`)

### Examples

Monitor `./images` folder with default settings:
```bash
python face_monitor.py ./images
```

Monitor with custom output file and check interval:
```bash
python face_monitor.py ./images --output results.txt --interval 2
```

Run on CPU:
```bash
python face_monitor.py ./images --device -1
```

## How It Works

1. The script monitors the specified folder for new images
2. When a new image is detected, it performs face recognition
3. Results are written to the output file with timestamp and recognized names
4. The monitor runs continuously until stopped with `Ctrl+C`

## Requirements

- Face database must be configured in `static/face_db.json`
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`
- GPU recommended for better performance
