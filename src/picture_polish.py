#!/usr/bin/env python3
"""
Picture Polish - Automatic Image Enhancement Tool

Supports: JPEG, HEIC, PNG, TIFF, WebP, BMP
Features: Auto contrast, sharpening, noise reduction, color optimization
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# OpenCV for advanced denoising
try:
    import cv2
    OPENCV_SUPPORT = True
except ImportError:
    OPENCV_SUPPORT = False


SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.tiff', '.tif', '.webp', '.bmp'}


class ImagePolisher:
    """Automatic image enhancement engine."""

    def __init__(self,
                 sharpen: float = 1.3,
                 contrast: float = 1.1,
                 color: float = 1.1,
                 brightness: float = 1.05,
                 denoise: bool = True,
                 auto_levels: bool = True,
                 autocut: bool = False):
        """
        Initialize the polisher with enhancement parameters.

        Args:
            sharpen: Sharpening factor (1.0 = no change, >1 = sharper)
            contrast: Contrast factor (1.0 = no change, >1 = more contrast)
            color: Color saturation factor (1.0 = no change, >1 = more vivid)
            brightness: Brightness factor (1.0 = no change, >1 = brighter)
            denoise: Apply noise reduction
            auto_levels: Apply automatic level adjustment
            autocut: Automatically detect and crop to document boundaries
        """
        self.sharpen = sharpen
        self.contrast = contrast
        self.color = color
        self.brightness = brightness
        self.denoise = denoise
        self.auto_levels = auto_levels
        self.autocut = autocut

    def load_image(self, path: Path) -> Image.Image:
        """Load an image from file."""
        suffix = path.suffix.lower()

        if suffix in {'.heic', '.heif'} and not HEIC_SUPPORT:
            raise RuntimeError("HEIC support requires pillow-heif. Install with: pip install pillow-heif")

        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {suffix}. Supported: {', '.join(SUPPORTED_FORMATS)}")

        img = Image.open(path)

        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if img.mode in ('RGBA', 'LA'):
            # Preserve alpha for formats that support it
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'RGBA':
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img, mask=img.split()[1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def auto_level(self, img: Image.Image) -> Image.Image:
        """Apply automatic level adjustment (histogram stretching)."""
        return ImageOps.autocontrast(img, cutoff=0.5)

    def apply_sharpening(self, img: Image.Image) -> Image.Image:
        """Apply unsharp mask for intelligent sharpening."""
        if self.sharpen <= 1.0:
            return img

        # Use unsharp mask for better quality sharpening
        enhancer = ImageEnhance.Sharpness(img)
        return enhancer.enhance(self.sharpen)

    def apply_contrast(self, img: Image.Image) -> Image.Image:
        """Enhance contrast."""
        if self.contrast == 1.0:
            return img
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(self.contrast)

    def apply_color(self, img: Image.Image) -> Image.Image:
        """Enhance color saturation."""
        if self.color == 1.0:
            return img
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(self.color)

    def apply_brightness(self, img: Image.Image) -> Image.Image:
        """Adjust brightness."""
        if self.brightness == 1.0:
            return img
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.brightness)

    def apply_denoise(self, img: Image.Image) -> Image.Image:
        """Apply noise reduction."""
        if not self.denoise:
            return img

        if OPENCV_SUPPORT:
            # Use OpenCV's non-local means denoising for best quality
            img_array = np.array(img)
            denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 6, 6, 7, 21)
            return Image.fromarray(denoised)
        else:
            # Fallback: mild blur for basic noise reduction
            return img.filter(ImageFilter.MedianFilter(size=3))

    def apply_autocut(self, img: Image.Image) -> Image.Image:
        """
        Automatically detect and crop to document/card boundaries.
        Uses edge detection and contour finding to locate the document.
        """
        if not self.autocut:
            return img

        if not OPENCV_SUPPORT:
            print("Warning: Autocut requires OpenCV. Skipping.", file=sys.stderr)
            return img

        img_array = np.array(img)
        original_height, original_width = img_array.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges to close gaps
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return img

        # Find the largest rectangular contour
        best_contour = None
        max_area = 0
        min_area_threshold = (original_width * original_height) * 0.1  # At least 10% of image

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_threshold:
                continue

            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Look for quadrilaterals (4 corners = document/card)
            if len(approx) == 4 and area > max_area:
                max_area = area
                best_contour = approx

        if best_contour is None:
            # Fallback: use bounding rectangle of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Add small padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(original_width - x, w + 2 * padding)
            h = min(original_height - y, h + 2 * padding)

            return img.crop((x, y, x + w, y + h))

        # Get the corner points and apply perspective transform
        pts = best_contour.reshape(4, 2)

        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = self._order_points(pts)

        # Calculate the width and height of the new image
        width_a = np.linalg.norm(rect[2] - rect[3])
        width_b = np.linalg.norm(rect[1] - rect[0])
        max_width = int(max(width_a, width_b))

        height_a = np.linalg.norm(rect[1] - rect[2])
        height_b = np.linalg.norm(rect[0] - rect[3])
        max_height = int(max(height_a, height_b))

        # Destination points for perspective transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        # Compute perspective transform and apply
        matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
        warped = cv2.warpPerspective(img_array, matrix, (max_width, max_height))

        return Image.fromarray(warped)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points as: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left has smallest sum, bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has smallest difference, bottom-left has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def analyze_image(self, img: Image.Image) -> dict:
        """Analyze image to suggest optimal enhancements."""
        img_array = np.array(img)

        # Calculate statistics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)

        suggestions = {
            'brightness_adjust': 1.0,
            'contrast_adjust': 1.0,
        }

        # Suggest brightness adjustment
        if brightness < 100:
            suggestions['brightness_adjust'] = 1.1 + (100 - brightness) / 200
        elif brightness > 180:
            suggestions['brightness_adjust'] = 0.95

        # Suggest contrast adjustment
        if contrast < 50:
            suggestions['contrast_adjust'] = 1.2
        elif contrast > 80:
            suggestions['contrast_adjust'] = 1.0

        return suggestions

    def polish(self, img: Image.Image, auto_adjust: bool = True) -> Image.Image:
        """
        Apply all enhancements to polish the image.

        Args:
            img: Input PIL Image
            auto_adjust: Automatically adjust parameters based on image analysis

        Returns:
            Polished PIL Image
        """
        # Analyze and auto-adjust if enabled
        if auto_adjust:
            analysis = self.analyze_image(img)
            original_brightness = self.brightness
            original_contrast = self.contrast
            self.brightness *= analysis['brightness_adjust']
            self.contrast *= analysis['contrast_adjust']

        # Apply enhancements in optimal order
        result = img.copy()

        # 0. Autocut first (crop to document boundaries)
        result = self.apply_autocut(result)

        # 1. Denoise first (reduces artifacts before other operations)
        result = self.apply_denoise(result)

        # 2. Auto levels
        if self.auto_levels:
            result = self.auto_level(result)

        # 3. Brightness adjustment
        result = self.apply_brightness(result)

        # 4. Contrast enhancement
        result = self.apply_contrast(result)

        # 5. Color saturation
        result = self.apply_color(result)

        # 6. Sharpening last (after all other adjustments)
        result = self.apply_sharpening(result)

        # Restore original settings if auto-adjusted
        if auto_adjust:
            self.brightness = original_brightness
            self.contrast = original_contrast

        return result

    def process_file(self, input_path: Path, output_path: Optional[Path] = None,
                     quality: int = 95, auto_adjust: bool = True) -> Path:
        """
        Process a single image file.

        Args:
            input_path: Path to input image
            output_path: Path for output (default: adds '_polished' suffix)
            quality: JPEG quality (1-100)
            auto_adjust: Automatically adjust parameters based on image

        Returns:
            Path to the output file
        """
        input_path = Path(input_path)

        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_polished.jpg"
        else:
            output_path = Path(output_path)

        # Load and process
        img = self.load_image(input_path)
        polished = self.polish(img, auto_adjust=auto_adjust)

        # Determine output format from extension
        suffix = output_path.suffix.lower()
        save_kwargs = {}

        if suffix in {'.jpg', '.jpeg'}:
            save_kwargs = {'quality': quality, 'optimize': True}
        elif suffix == '.png':
            save_kwargs = {'optimize': True}
        elif suffix == '.webp':
            save_kwargs = {'quality': quality, 'method': 6}

        polished.save(output_path, **save_kwargs)
        return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Picture Polish - Automatic Image Enhancement Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python picture_polish.py photo.jpg                    # Basic polish
  python picture_polish.py photo.HEIC -o enhanced.jpg   # HEIC to polished JPEG
  python picture_polish.py image.png --sharpen 1.5      # Extra sharpening
  python picture_polish.py photo.jpg --preset vivid     # Use vivid preset
        '''
    )

    parser.add_argument('input', help='Input image file')
    parser.add_argument('-o', '--output', help='Output file path (default: input_polished.jpg)')
    parser.add_argument('-q', '--quality', type=int, default=95, help='Output quality 1-100 (default: 95)')

    # Enhancement parameters
    parser.add_argument('--sharpen', type=float, default=1.3, help='Sharpening factor (default: 1.3)')
    parser.add_argument('--contrast', type=float, default=1.1, help='Contrast factor (default: 1.1)')
    parser.add_argument('--color', type=float, default=1.1, help='Color saturation (default: 1.1)')
    parser.add_argument('--brightness', type=float, default=1.05, help='Brightness factor (default: 1.05)')

    # Toggles
    parser.add_argument('--no-denoise', action='store_true', help='Disable noise reduction')
    parser.add_argument('--no-auto-levels', action='store_true', help='Disable auto level adjustment')
    parser.add_argument('--no-auto-adjust', action='store_true', help='Disable automatic parameter adjustment')
    parser.add_argument('--autocut', action='store_true',
                        help='Auto-detect and crop to document/card boundaries (for documents, IDs, passports)')

    # Presets
    parser.add_argument('--preset', choices=['subtle', 'balanced', 'vivid', 'sharp', 'document', 'passport', 'id'],
                        help='Use a preset configuration')

    args = parser.parse_args()

    # Apply presets
    if args.preset:
        presets = {
            # Photo presets
            'subtle': {'sharpen': 1.1, 'contrast': 1.05, 'color': 1.05, 'brightness': 1.02},
            'balanced': {'sharpen': 1.3, 'contrast': 1.1, 'color': 1.1, 'brightness': 1.05},
            'vivid': {'sharpen': 1.4, 'contrast': 1.2, 'color': 1.3, 'brightness': 1.1},
            'sharp': {'sharpen': 1.6, 'contrast': 1.15, 'color': 1.1, 'brightness': 1.05},
            # Document presets - optimized for text clarity
            'document': {'sharpen': 1.8, 'contrast': 1.4, 'color': 0.9, 'brightness': 1.15},
            'passport': {'sharpen': 1.4, 'contrast': 1.2, 'color': 1.0, 'brightness': 1.05},
            'id': {'sharpen': 1.5, 'contrast': 1.3, 'color': 0.95, 'brightness': 1.1},
        }
        preset = presets[args.preset]
        args.sharpen = preset['sharpen']
        args.contrast = preset['contrast']
        args.color = preset['color']
        args.brightness = preset['brightness']

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"Error: Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}", file=sys.stderr)
        sys.exit(1)

    # Create polisher
    polisher = ImagePolisher(
        sharpen=args.sharpen,
        contrast=args.contrast,
        color=args.color,
        brightness=args.brightness,
        denoise=not args.no_denoise,
        auto_levels=not args.no_auto_levels,
        autocut=args.autocut
    )

    # Process
    try:
        output_path = args.output
        if output_path:
            output_path = Path(output_path)

        result = polisher.process_file(
            input_path,
            output_path,
            quality=args.quality,
            auto_adjust=not args.no_auto_adjust
        )
        print(f"Polished image saved to: {result}")

    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
