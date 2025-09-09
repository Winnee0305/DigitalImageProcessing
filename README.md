# Digital Image Processing Project

A comprehensive video processing application that applies various digital image processing techniques to enhance and modify video content.

## Overview

This project implements a sophisticated video processing pipeline that applies multiple digital image processing techniques to enhance video content. The application processes various video files and applies effects including face detection and blurring, brightness adjustment, logo overlay, watermarking, and fade effects.

## Features

### 🎥 Video Processing Capabilities

- **Multi-video Support**: Process different video files (street, traffic, singapore, office)
- **Real-time Processing**: Frame-by-frame processing with progress tracking
- **Output Generation**: Creates processed videos in AVI format

### 🔍 Computer Vision Features

- **Face Detection & Blurring**:
  - Uses Haar Cascade classifier for face detection
  - Implements secondary face detection for improved accuracy
  - Applies Gaussian blur to detected faces for privacy protection
- **Brightness Adjustment**:
  - Automatic day/night detection based on mean intensity
  - Dynamic brightness enhancement for low-light conditions

### 🎨 Visual Effects

- **Logo Overlay**: Custom logo with gradient background
- **Watermarking**: Rotating watermarks that change every 5 seconds
- **Fade Effects**: Smooth fade-in and fade-out transitions
- **Narrator Integration**: Overlay talking video with chroma key masking

### 🛠️ Technical Features

- **Size Adjustment**: Automatic video resizing to standard dimensions (1280x720)
- **Frame Rate Management**: Handles different input frame rates
- **Memory Optimization**: Efficient frame processing with temporary storage

## Prerequisites

### Required Software

- **Python 3.x**
- **Anaconda** (recommended)
- **Spyder IDE** (included with Anaconda)

### Required Libraries

```bash
pip install opencv-python
pip install numpy
pip install matplotlib
```

### Required Files

- `face_detector.xml` - Haar Cascade classifier for face detection
- `talking.mp4` - Narrator video file
- `endscreen.mp4` - End screen video
- `watermark1.png` - First watermark image
- `watermark2.png` - Second watermark image

## Installation & Setup

1. **Download/Clone the project**

   ```bash
   git clone <repository-url>
   cd DigitalImageProcessing
   ```

2. **Install Anaconda** (if not already installed)

   - Download from [Anaconda website](https://www.anaconda.com/products/distribution)
   - Follow installation instructions for your operating system

3. **Install required Python packages**

   ```bash
   conda install opencv numpy matplotlib
   # or
   pip install opencv-python numpy matplotlib
   ```

4. **Verify all required files are present**
   - Ensure all video files (.mp4) are in the project directory
   - Verify `face_detector.xml` is present
   - Check that watermark images (.png) are available

## Usage

### Running the Application

1. **Launch Anaconda and open Spyder**
2. **Open the script file** `DIP_Code.py` in Spyder
3. **Run the script** by clicking the "Run" button or pressing F5
4. **Select video to process** from the menu:
   - Enter `1` for 'street' video
   - Enter `2` for 'traffic' video
   - Enter `3` for 'singapore' video
   - Enter `4` for 'office' video
   - Enter `Q` to exit

### Output

- Processed videos are saved as `processed_<original_name>.avi`
- Videos are processed at 30 FPS with 1280x720 resolution
- Processing progress is displayed in the console

## Project Structure

```
DigitalImageProcessing/
├── DIP_Code.py              # Main processing script
├── face_detector.xml        # Haar Cascade classifier
├── talking.mp4              # Narrator video
├── endscreen.mp4            # End screen video
├── watermark1.png           # First watermark
├── watermark2.png           # Second watermark
├── street.mp4               # Sample video 1
├── traffic.mp4              # Sample video 2
├── singapore.mp4            # Sample video 3
├── office.mp4               # Sample video 4
├── DIP_Report.pdf           # Project documentation
└── README.md                # This file
```

## Key Functions

### Core Processing Functions

- `processVideo()` - Main video processing pipeline
- `adjustBrightness()` - Automatic brightness adjustment
- `faceDetection()` - Face detection using Haar Cascade
- `faceBlurring()` - Apply blur effect to detected faces
- `addLogo()` - Overlay custom logo with gradient
- `addWatermark()` - Apply rotating watermarks
- `applyFadeEffect()` - Apply fade in/out transitions

### Utility Functions

- `resize()` - Image/video resizing with positioning
- `mergeBackgroundnForeground()` - Image compositing
- `extractMaskByPeakColor()` - Chroma key masking
- `createLogo()` - Generate custom logo
- `createGradient()` - Create gradient backgrounds

## Technical Details

### Video Processing Pipeline

1. **Initialization**: Load video files and set up processing parameters
2. **Frame Processing**: Process each frame individually
3. **Brightness Adjustment**: Analyze and adjust frame brightness
4. **Face Detection**: Detect faces using multi-frame analysis
5. **Face Blurring**: Apply privacy protection to detected faces
6. **Size Adjustment**: Resize to standard dimensions if needed
7. **Narrator Overlay**: Add talking video with chroma key
8. **Logo Addition**: Overlay custom logo with gradient
9. **Watermarking**: Apply rotating watermarks
10. **Fade Effects**: Apply smooth transitions
11. **Output**: Write processed frame to output video

### Performance Considerations

- **Memory Management**: Efficient frame processing with temporary storage
- **Progress Tracking**: Real-time processing status updates
- **Error Handling**: Robust error handling for video processing
- **Optimization**: Optimized algorithms for real-time processing

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all required libraries are installed
2. **File Not Found**: Verify all video and image files are in the correct directory
3. **Memory Issues**: For large videos, ensure sufficient RAM is available
4. **OpenCV Issues**: Update OpenCV to the latest version if face detection fails

### Performance Tips

- Close other applications to free up memory
- Use SSD storage for faster video I/O
- Process videos in smaller chunks if memory is limited

**Note**: This application is designed for educational and research purposes. Ensure you have proper rights to process any video content.
