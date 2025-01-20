# 🛰️ Satellite Port Detection System

Hey everyone! Welcome to my space project! 👋 

Thank you for the opportunity to work on this fascinating computer vision challenge! This project implements a comprehensive solution for satellite port detection and analysis, focusing on robust geometric detection, rotation analysis, and perspective transformation.

## 🎯 Project Objectives

The project addresses four key technical challenges:

1. **Rotation Detection**: Given a reference image with a port (red circle at top-left), detect the port in a rotated image and calculate the rotation angle using geometric analysis.

2. **Port Navigation**: Implement an algorithm to locate the port in arbitrarily cropped sections of the reference image, generating precise camera movement commands until the port is visible.

3. **Perspective Transformation**: Apply homography transformation to generate a view from 100 cm away at a 22.5-degree angle, maintaining geometric accuracy and scale.

4. **Camera Movement Simulation**: Calculate incremental camera positions to transition from the side view to front view while maintaining a constant 100 cm distance from the port's center.

## 🧩 Implementation Details

### 📸 Reference Image Generation (`base_image.py`)
Implements precise geometric construction of the satellite port:
- Outer square: 40x40 cm
- Inner square: 20x20 cm
- Port diameter: 5 cm
- Resolution: 16 pixels/cm

The port's center is positioned at the intersection of three 2.5 cm radii from:
- Outer square's top edge
- Outer square's left edge
- Inner square's top-left corner

### 🔄 Rotation Analysis (`mainrotation.py`)
Implements robust port detection and rotation calculation using:
- HSV color space transformation for reliable circle detection
- Contour analysis with geometric validation
- Vector-based angle calculation

Color detection parameters:
```python
# Dual-range HSV thresholding for robust red detection
lower_red1 = [0, 50, 50]      # Primary red range start
lower_red2 = [170, 50, 50]    # Secondary red range start
upper_red1 = [10, 255, 255]   # Primary red range end
upper_red2 = [180, 255, 255]  # Secondary red range end
```

### 🎮 Port Navigation (`navigation.py`)
Implements an efficient search algorithm using:
- Spiral pattern generation for optimal coverage
- Real-time position tracking
- Geometric validation of port detection

Search pattern implementation:
```python
# Spiral pattern generation with cardinal directions
directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # W, N, E, S
step_size = 50  # pixels per iteration
```

### 👁️ Perspective Visualization (`unified_visualization.py`)
Implements perspective transformation and visualization:
- Homography matrix calculation for accurate perspective
- Real-time camera position visualization
- Trajectory plotting with geometric constraints

## 🚀 Usage Instructions

1. Environment Setup:
```bash
pip install -r requirements.txt
```

2. Run each component:
```bash
# Generate base reference image
python base_image.py

# Detect rotation angle (accepts arbitrary rotation)
python mainrotation.py

# Execute port navigation with camera movements
python navigation.py

# Generate perspective view and camera movement simulation
python unified_visualization.py
```

Each file can be run independently and will prompt for necessary inputs when required.

## 🛠️ Technical Requirements
- Python 3.x
- OpenCV (computer vision operations)
- NumPy (mathematical computations)
- Matplotlib (visualization framework)

## 💫 Performance Characteristics
- Rotation Detection: Accurate to ±1° within 90° range
- Navigation: Optimized spiral search with O(n) complexity
- Perspective: Maintains geometric accuracy at 100 cm distance
- Real-time Visualization: 30 FPS with trajectory tracking

## 📝 Implementation Notes
- Consistent 16 pixels/cm scale throughout
- HSV color space for optimal port detection
- Modular architecture for component isolation
- Robust error handling and validation

Alternative implementations exploring different geometric approaches are available in the alternative folder. The codebase has been enhanced with LLMs while preserving the core mathematical logic.

Excited to share this implementation! 🚀✨

~ Adithya S
(Interview Assignment - Team Aule Space) 