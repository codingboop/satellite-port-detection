# 🛰️ Satellite Port Detection System

Hey there! 👋 Welcome to my Satellite Port Detection project. This is a cool computer vision system I built that can find, analyze, and navigate to a satellite port. Think of it as teaching a camera to find a specific target (a red circle surrounded by squares) and figure out how it's positioned.

## 🎯 What Does It Do?

This project helps solve three main challenges:
1. Finding a red circle within squares (like finding a specific port on a satellite)
2. Figuring out if the port is rotated (and by how much)
3. Moving a camera around to find the port when it's partially hidden

## 🧩 Main Parts

### 📸 Creating the Reference Image (`base_image.py`)
This is where we create our test image. Imagine drawing two squares (one big, one small) with a red circle that touches specific points on both squares. The measurements are pretty straightforward:
- Big square: 40x40 cm
- Small square: 20x20 cm
- Red circle: 5 cm across
- Each centimeter is 16 pixels in the image

The tricky part was getting the circle in just the right spot - it needs to be exactly 2.5 cm from the edges and corners. (Trust me, this took some trial and error to get right! 😅)

### 🔄 Detecting Rotation (`mainrotation.py`)
This part figures out how much the port is rotated. It's like when you're trying to read text that's tilted - your brain automatically figures out the angle. Here, we're teaching the computer to do the same thing.

The cool stuff it does:
- Finds the red circle using color detection (turns out HSV color space works great for this!)
- Calculates the exact angle of rotation
- Tells us how confident it is about its answer

Here's a peek at how we detect the red color:
```python
# We look for red in two ranges because red appears at both ends of the HSV spectrum
lower_red1 = [0, 50, 50]      # Start of red spectrum
lower_red2 = [170, 50, 50]    # End of red spectrum
upper_red1 = [10, 255, 255]   # Bright reds
upper_red2 = [180, 255, 255]  # Dark reds
```

### 🎮 Camera Navigation (`navigation.py`)
This is probably the most fun part! Imagine you're playing a video game where you need to find something, but you can only see a small part of the map at a time. That's exactly what this does - it moves a virtual camera around in a spiral pattern until it spots the port.

How it searches:
```python
# The camera moves in this pattern:
#  → ↑ ← ↓ (and repeat, getting bigger each time)
directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
step_size = 50  # pixels per move
```

### 👁️ Visualization (`unified_visualization.py`)
This brings everything together with some nice visuals. You can:
- See the port from different angles (0° to 90°)
- Watch the camera move around as it searches
- See a 3D-like view of how the port looks from different positions

## 🚀 Want to Try It?

1. First, install the stuff you need:
```bash
pip install -r requirements.txt
```

2. Then you can play with each part:
```python
# Make a test image
from base_image import SatellitePort
port = SatellitePort()
image = port.generate_image()

# Test rotation detection (22.5° is our standard test angle)
from mainrotation import EnhancedRotationDetector
detector = EnhancedRotationDetector()
angle = detector.process_image(22.5)

# Try the navigation
from navigation import CameraNavigator
navigator = CameraNavigator()
found, commands = navigator.search_for_circle()

# See it all in action
from unified_visualization import UnifiedVisualizer
visualizer = UnifiedVisualizer()
visualizer.visualize_unified(angle=22.5)
```

## 🛠️ What You Need
- Python 3.x
- OpenCV (for image stuff)
- NumPy (for math)
- Matplotlib (for making it look pretty)

## 💫 Cool Features
- Works really well for angles up to 90°
- Usually finds the port in a few seconds
- Shows you exactly how it's thinking with real-time visuals
- Keeps the camera 100 cm from the port (just like a real satellite might!)

## 📝 Good to Know
- Everything's measured in centimeters (1 cm = 16 pixels)
- We use HSV colors because they're more reliable than RGB for finding specific colors
- The code's organized into separate files so it's easy to understand and modify
- There's plenty of error checking so it won't crash on you

## 🤝 Contributing
Found a bug? Have an idea to make it better? Feel free to open an issue or send a pull request! I'm always happy to improve things.

You can reach me at:
- GitHub: [@codingboop](https://github.com/codingboop)

## 📜 License
This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 codingboop

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``` 