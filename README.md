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
- **Document Mode** - Specialized presets for documents, passports, and IDs
- **Auto-Cut** - Automatic document boundary detection and perspective correction

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

### Document, Passport & ID Processing

```bash
# Scanned document - high contrast, sharp text
python src/picture_polish.py scan.jpg --preset document

# Passport photo - natural skin tones, balanced
python src/picture_polish.py passport.jpg --preset passport

# ID card - optimized for small text and details
python src/picture_polish.py id_card.jpg --preset id

# Auto-cut: detect and crop to document boundaries
python src/picture_polish.py document_photo.jpg --preset document --autocut

# Auto-cut also corrects perspective (skewed documents)
python src/picture_polish.py tilted_id.jpg --preset id --autocut
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
| `--preset` | - | Use preset (see table below) |
| `--autocut` | - | Auto-detect and crop to document boundaries |
| `--no-denoise` | - | Disable noise reduction |
| `--no-auto-levels` | - | Disable auto level adjustment |
| `--no-auto-adjust` | - | Disable adaptive parameter adjustment |

## Presets

### Photo Presets

| Preset | Sharpen | Contrast | Color | Brightness | Use Case |
|--------|---------|----------|-------|------------|----------|
| subtle | 1.1 | 1.05 | 1.05 | 1.02 | Light touch-up |
| balanced | 1.3 | 1.1 | 1.1 | 1.05 | General purpose |
| vivid | 1.4 | 1.2 | 1.3 | 1.1 | Bold, vibrant photos |
| sharp | 1.6 | 1.15 | 1.1 | 1.05 | Maximum clarity |

### Document Presets

| Preset | Sharpen | Contrast | Color | Brightness | Use Case |
|--------|---------|----------|-------|------------|----------|
| document | 1.8 | 1.4 | 0.9 | 1.15 | Scans, receipts, text |
| passport | 1.4 | 1.2 | 1.0 | 1.05 | Passport photos |
| id | 1.5 | 1.3 | 0.95 | 1.1 | ID cards, licenses |

*Document presets reduce color saturation for cleaner text and increase contrast for readability.*

## Supported Formats

**Input:** JPEG, HEIC, HEIF, PNG, TIFF, WebP, BMP

**Output:** JPEG, PNG, WebP (determined by output file extension)

## How It Works

1. **Load** - Read image and convert to RGB
2. **Auto-cut** (optional) - Detect document edges, crop and correct perspective
3. **Analyze** - Calculate brightness/contrast statistics
4. **Denoise** - Apply noise reduction filter
5. **Auto-levels** - Stretch histogram for full dynamic range
6. **Brightness** - Adjust overall luminance
7. **Contrast** - Enhance tonal separation
8. **Color** - Adjust saturation
9. **Sharpen** - Apply unsharp mask for clarity

### Auto-Cut Feature

The `--autocut` flag uses computer vision to:
- Detect document/card boundaries using edge detection
- Find the largest rectangular contour
- Apply perspective transformation to correct skewed images
- Crop to the document boundaries

Works best with documents, IDs, passports, and cards photographed on contrasting backgrounds.

## License

MIT License
