# Picture Polish

A command-line tool for automatically enhancing and optimizing images. Supports JPEG, HEIC, PNG, TIFF, WebP, and BMP formats.

## Features

- **Auto Contrast** - Histogram stretching for optimal dynamic range
- **Sharpening** - Intelligent unsharp mask for crisp details
- **Noise Reduction** - OpenCV non-local means denoising (with fallback)
- **Color Enhancement** - Saturation boost for vivid images
- **Brightness Optimization** - Adaptive brightness adjustment
- **Smart Analysis** - Automatically adjusts parameters based on image characteristics
- **Presets** - Quick settings for different enhancement styles

## Installation

```bash
# Clone the repository
git clone https://github.com/qwitch13/picture-polish.git
cd picture-polish

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **Pillow** - Core image processing
- **pillow-heif** - HEIC/HEIF format support (Apple photos)
- **NumPy** - Array operations
- **OpenCV** - Advanced noise reduction (optional but recommended)

## Usage

### Basic Usage

```bash
# Polish an image with default settings
python src/picture_polish.py photo.jpg

# Specify output file
python src/picture_polish.py photo.jpg -o enhanced.jpg

# Convert HEIC to polished JPEG
python src/picture_polish.py IMG_1234.HEIC -o polished.jpg
```

### Using Presets

```bash
# Subtle - Light touch-up
python src/picture_polish.py photo.jpg --preset subtle

# Balanced - Default enhancement (same as no preset)
python src/picture_polish.py photo.jpg --preset balanced

# Vivid - Bold colors and contrast
python src/picture_polish.py photo.jpg --preset vivid

# Sharp - Maximum sharpening
python src/picture_polish.py photo.jpg --preset sharp
```

### Custom Parameters

```bash
# Extra sharpening
python src/picture_polish.py photo.jpg --sharpen 1.5

# High contrast with vivid colors
python src/picture_polish.py photo.jpg --contrast 1.3 --color 1.4

# Brighten a dark photo
python src/picture_polish.py dark_photo.jpg --brightness 1.2

# High quality output
python src/picture_polish.py photo.jpg -q 100
```

### Disabling Features

```bash
# Skip noise reduction (faster processing)
python src/picture_polish.py photo.jpg --no-denoise

# Skip auto-levels
python src/picture_polish.py photo.jpg --no-auto-levels

# Use fixed parameters (no adaptive adjustment)
python src/picture_polish.py photo.jpg --no-auto-adjust
```

## Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `{input}_polished.jpg` | Output file path |
| `-q, --quality` | `95` | JPEG quality (1-100) |
| `--sharpen` | `1.3` | Sharpening factor (1.0 = no change) |
| `--contrast` | `1.1` | Contrast factor (1.0 = no change) |
| `--color` | `1.1` | Color saturation (1.0 = no change) |
| `--brightness` | `1.05` | Brightness factor (1.0 = no change) |
| `--preset` | - | Use preset: `subtle`, `balanced`, `vivid`, `sharp` |
| `--no-denoise` | - | Disable noise reduction |
| `--no-auto-levels` | - | Disable auto level adjustment |
| `--no-auto-adjust` | - | Disable adaptive parameter adjustment |

## Presets

| Preset | Sharpen | Contrast | Color | Brightness |
|--------|---------|----------|-------|------------|
| subtle | 1.1 | 1.05 | 1.05 | 1.02 |
| balanced | 1.3 | 1.1 | 1.1 | 1.05 |
| vivid | 1.4 | 1.2 | 1.3 | 1.1 |
| sharp | 1.6 | 1.15 | 1.1 | 1.05 |

## Supported Formats

**Input:** JPEG, HEIC, HEIF, PNG, TIFF, WebP, BMP

**Output:** JPEG, PNG, WebP (determined by output file extension)

## How It Works

1. **Load** - Read image and convert to RGB
2. **Analyze** - Calculate brightness/contrast statistics
3. **Denoise** - Apply noise reduction filter
4. **Auto-levels** - Stretch histogram for full dynamic range
5. **Brightness** - Adjust overall luminance
6. **Contrast** - Enhance tonal separation
7. **Color** - Boost saturation
8. **Sharpen** - Apply unsharp mask for clarity

## License

MIT License
