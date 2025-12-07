# Picture Polish - Development Journal

## Project Overview
An image polishing tool that automatically enhances and optimizes images (JPEG, HEIC, PNG, etc.) to make them clearer and more visually appealing.

## Progress Log

### Session 1 - Initial Development

**Goal:** Create a complete image polishing program with auto-optimization features.

**Planned Features:**
- Support for multiple image formats (JPEG, HEIC, PNG, TIFF, WebP)
- Auto contrast enhancement
- Sharpening
- Noise reduction
- Color optimization
- Brightness/saturation adjustment
- CLI interface for easy usage

**Tech Stack:**
- Python 3
- Pillow (PIL) - Core image processing
- pillow-heif - HEIC format support
- NumPy - Array operations for advanced filters
- OpenCV (optional) - Advanced denoising

---

**Status:** Implementation complete!

**Completed:**
1. Created `requirements.txt` with dependencies (Pillow, pillow-heif, numpy, opencv-python)
2. Built `src/picture_polish.py` with:
   - `ImagePolisher` class for all enhancement operations
   - Support for JPEG, HEIC, PNG, TIFF, WebP, BMP formats
   - Auto contrast/levels adjustment
   - Intelligent sharpening
   - Noise reduction (OpenCV non-local means or fallback median filter)
   - Color saturation enhancement
   - Brightness adjustment
   - Image analysis for auto-adjusting parameters
   - Presets: subtle, balanced, vivid, sharp
3. Full CLI interface with all options exposed

**Usage:**
```bash
# Install dependencies
pip install -r requirements.txt

# Basic usage
python src/picture_polish.py photo.jpg

# HEIC to polished JPEG
python src/picture_polish.py photo.HEIC -o enhanced.jpg

# With vivid preset
python src/picture_polish.py image.png --preset vivid
```

---

### Update 1 - Documentation & Cleanup

- Added comprehensive README.md with:
  - Installation instructions
  - Usage examples
  - Full options reference table
  - Preset comparison table
  - Processing pipeline explanation
- Fixed .gitignore to exclude Python __pycache__ files
- Pushed to GitHub: https://github.com/qwitch13/picture-polish

---

### Update 2 - Document Presets & Auto-Cut

**New Presets Added:**
- `document` - Optimized for scanned documents, receipts, text (high sharpening 1.8, high contrast 1.4, reduced color 0.9)
- `passport` - For passport photos (balanced settings, natural skin tones)
- `id` - For ID cards and licenses (sharp text, good contrast)

**New Feature: Auto-Cut (`--autocut`)**
- Automatic document boundary detection using edge detection and contour finding
- Perspective correction for skewed/tilted documents
- Crops precisely to document edges
- Works with documents, IDs, passports, business cards

**Usage:**
```bash
# Document with auto-crop
python src/picture_polish.py document_photo.jpg --preset document --autocut

# ID card processing
python src/picture_polish.py id_scan.jpg --preset id --autocut
```
