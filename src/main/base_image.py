import numpy as np
import cv2
import matplotlib.pyplot as plt

class SatellitePort:
    def __init__(self, scale_factor=10):
        """
        Initialize the satellite port with measurements
        scale_factor: pixels per cm
        """
        self.scale_factor = scale_factor
        
        # Measurements in cm
        self.outer_square_size = 40
        self.inner_square_size = 30
        self.spacing = 5  # Fixed spacing between squares
        
        # Convert to pixels
        self.outer_size_px = int(self.outer_square_size * scale_factor)
        self.inner_size_px = int(self.inner_square_size * scale_factor)
        self.spacing_px = int(self.spacing * scale_factor)
        
        # Create base image
        self.image = np.ones((self.outer_size_px, self.outer_size_px, 3), dtype=np.uint8) * 255
        
    def draw_squares(self):
        """Draw the outer and inner squares"""
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
        """
        Calculate the radius of the circle that touches:
        1. Top edge of outer square
        2. Left edge of outer square
        3. Top-left corner of inner square
        
        The circle's center must be equidistant from:
        - The top edge of outer square (y=0)
        - The left edge of outer square (x=0)
        - The point (spacing_px, spacing_px) which is the inner square's corner
        """
        # The circle's center will be (r,r) where r is the radius
        # For the circle to touch the inner square corner at (spacing_px, spacing_px),
        # we use the distance formula: r = sqrt((x2-x1)^2 + (y2-y1)^2)
        # where (x1,y1) is the center (r,r) and (x2,y2) is (spacing_px, spacing_px)
        
        # Let's solve: r = sqrt((spacing_px-r)^2 + (spacing_px-r)^2)
        # r = sqrt(2(spacing_px-r)^2)
        # r = |spacing_px-r| * sqrt(2)
        # r = spacing_px * sqrt(2) - r * sqrt(2)
        # r + r * sqrt(2) = spacing_px * sqrt(2)
        # r * (1 + sqrt(2)) = spacing_px * sqrt(2)
        # r = (spacing_px * sqrt(2)) / (1 + sqrt(2))
        
        radius = int((self.spacing_px * np.sqrt(2)) / (1 + np.sqrt(2)))
        return radius
    
    def draw_circle(self):
        """Draw the circular marker at top-left"""
        radius = self.calculate_circle_radius()
        # Circle center is radius distance from both top and left edges
        center = (radius, radius)
        cv2.circle(self.image, center, radius, (0, 0, 255), 2)
        
        # For verification, draw a small point at inner square's top-left corner
        inner_corner = (self.spacing_px, self.spacing_px)
        cv2.circle(self.image, inner_corner, 2, (255, 0, 0), -1)
    
    def generate_image(self):
        """Generate the complete satellite port image"""
        self.draw_squares()
        self.draw_circle()
        return self.image
    
    def save_image(self, filename='satellite_port.png'):
        """Save the generated image"""
        cv2.imwrite(filename, self.image)
        
    def display_image(self):
        """Display the image using matplotlib"""
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Satellite Port (Red: Circle, Blue: Inner Square Corner)')
        plt.show()

def main():
    # Create satellite port with scale factor (10 pixels per cm)
    port = SatellitePort(scale_factor=10)
    
    # Generate and save the image
    port.generate_image()
    port.save_image()
    port.display_image()

if __name__ == "__main__":
    main() 