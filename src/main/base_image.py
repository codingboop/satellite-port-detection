import numpy as np
import cv2
import matplotlib.pyplot as plt

class SatellitePort:
    def __init__(self, scale_factor=10):
        # scale_factor: pixels per cm
        self.scale_factor = scale_factor
        
        # Measurements in cm
        self.outer_square_size = 40
        self.inner_square_size = 30
        self.spacing = 5  # Fixed spacing between squares
        
        # Convert to pixels
        self.outer_size_px = int(self.outer_square_size * scale_factor)
        self.inner_size_px = int(self.inner_square_size * scale_factor)
        self.spacing_px = int(self.spacing * scale_factor)
        
        self.image = np.ones((self.outer_size_px, self.outer_size_px, 3), dtype=np.uint8) * 255
        
    def draw_squares(self):
        # Outer square (black)
        cv2.rectangle(self.image, (0, 0), 
                     (self.outer_size_px-1, self.outer_size_px-1), 
                     (0, 0, 0), 2)
        
        # Inner square
        start_point = self.spacing_px
        cv2.rectangle(self.image,
                     (start_point, start_point),
                     (start_point + self.inner_size_px-1,
                      start_point + self.inner_size_px-1),
                     (0, 0, 0), 2)
    
    def calculate_circle_radius(self):
        # Calculate radius of circle touching:
        # 1. Top edge of outer square
        # 2. Left edge of outer square
        # 3. Top-left corner of inner square
        # Solution: r = (spacing_px * sqrt(2)) / (1 + sqrt(2))
        radius = int((self.spacing_px * np.sqrt(2)) / (1 + np.sqrt(2)))
        return radius
    
    def draw_circle(self):
        radius = self.calculate_circle_radius()
        center = (radius, radius)
        cv2.circle(self.image, center, radius, (0, 0, 255), 2)
        
        # Mark inner square's top-left corner for verification
        inner_corner = (self.spacing_px, self.spacing_px)
        cv2.circle(self.image, inner_corner, 2, (255, 0, 0), -1)
    
    def generate_image(self):
        self.draw_squares()
        self.draw_circle()
        return self.image
    
    def save_image(self, filename='satellite_port.png'):
        cv2.imwrite(filename, self.image)
        
    def display_image(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Satellite Port (Red: Circle, Blue: Inner Square Corner)')
        plt.show()

def main():
    port = SatellitePort(scale_factor=10)
    port.generate_image()
    port.save_image()
    port.display_image()

if __name__ == "__main__":
    main() 