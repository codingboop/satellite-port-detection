# 🛰️ Satellite Port Detection System

Hey everyone! Welcome to my space project! 👋 

Thank you for the amazing opportunity to work on this super cool project! I had a blast implementing a computer vision system that can detect, analyze, and navigate to a satellite port. It was such a fun exercise that really got me thinking about how we can use fundamental computer vision principles in space applications!

## 🎯 What's This All About?

I tackled three main challenges in this project:
1. Finding a red circle within squares (think of it as locating a specific port on a satellite)
2. Working out how much the port is rotated
3. Moving a camera around to find the port even when it's partially hidden

## 🧩 The Cool Parts

### 📸 Making the Reference Image (`base_image.py`)
First up, I created a test image with two squares and a red circle. The measurements are:
- Big square: 40x40 cm
- Small square: 20x20 cm
- Red circle: 5 cm across
- Each centimeter = 16 pixels

Getting the circle in exactly the right spot was tricky - it needs to be precisely 2.5 cm from the edges and corners. Took some trial and error, but got there in the end! 😅

### 🔄 Figuring Out Rotation (`mainrotation.py`)
This part was interesting! I taught the computer to figure out how much the port is rotated. You know how your brain can tell when text is tilted? I implemented something similar here!

The neat stuff it does:
- Spots the red circle using HSV color space (works way better than RGB!)
- Works out the exact rotation angle
- Tells you how sure it is about its answer

Here's a bit of the color detection magic:
```python
# Red appears at both ends of the HSV spectrum, so we need two ranges
lower_red1 = [0, 50, 50]      # Start of red spectrum
lower_red2 = [170, 50, 50]    # End of red spectrum
upper_red1 = [10, 255, 255]   # Bright reds
upper_red2 = [180, 255, 255]  # Dark reds
```

### 🎮 Finding the Port (`navigation.py`)
This was probably my favorite part! It's like a little game where the camera has to find the port. I made it move in a spiral pattern - pretty efficient way to search an area!

Check out how it moves:
```python
# The camera does this pattern:
#  → ↑ ← ↓ (and keeps going, getting bigger each time)
directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
step_size = 50  # pixels per move
```

### 👁️ Making It All Visual (`unified_visualization.py`)
This brings everything together with some cool visuals. You can:
- Look at the port from different angles (0° to 90°)
- Watch the camera searching around
- See how the port looks from different positions

## 🚀 Want to Give It a Try?

1. Get the basics installed:
```bash
pip install -r requirements.txt
```

2. Play around with it:
```python
# Create a test image
from base_image import SatellitePort
port = SatellitePort()
image = port.generate_image()

# Check out the rotation detection (22.5° is my test angle)
from mainrotation import EnhancedRotationDetector
detector = EnhancedRotationDetector()
angle = detector.process_image(22.5)

# Try the navigation system
from navigation import CameraNavigator
navigator = CameraNavigator()
found, commands = navigator.search_for_circle()

# See all the visuals
from unified_visualization import UnifiedVisualizer
visualizer = UnifiedVisualizer()
visualizer.visualize_unified(angle=22.5)
```

## 🛠️ What You'll Need
- Python 3.x
- OpenCV (for the computer vision stuff)
- NumPy (for all the math)
- Matplotlib (to make everything look nice)

## 💫 Some Cool Things About It
- Works great for angles up to 90°
- Finds the port pretty quickly
- Shows you exactly what it's doing
- Keeps the camera 100 cm from the port (just like a real satellite!)

## 📝 Quick Notes
- Everything's in centimeters (1 cm = 16 pixels)
- Used HSV colors because they're more reliable
- Code is nicely organized in different files
- Lots of error checking to keep things running smooth

I've also tried some alternative approaches which you can find in the alternative folder - feel free to explore those too! I used LLMs to help make the code more readable while keeping my core logic intact.

Have fun with it, and let me know how it goes for you! 🚀✨

~ Adithya S

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