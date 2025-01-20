import numpy as np
import cv2
import matplotlib.pyplot as plt
from base_image import SatellitePort

class PerspectiveTransformer:
    def __init__(self, distance_cm: float = 100.0, angle_deg: float = 22.5):
        """Initialize transformer with given parameters.
        Args:
            distance_cm: Distance from camera to port center (default: 100.0 cm)
            angle_deg: Viewing angle from right side (default: 22.5 degrees)
                      Can be any angle between 0 (front view) and 90 (side view)
        """
        port = SatellitePort()
        self.front_image = port.generate_image()
        self.height, self.width = self.front_image.shape[:2]
        
        if not 0 <= angle_deg <= 90:
            raise ValueError("Angle must be between 0 and 90 degrees")
        if distance_cm <= 0:
            raise ValueError("Distance must be positive")
            
        self.distance_cm = distance_cm
        self.angle_rad = np.deg2rad(angle_deg)
        
        hsv = cv2.cvtColor(self.front_image, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            self.port_center = (int(x), int(y))
            self.port_radius = int(radius)
        else:
            self.port_center = (self.width // 2, self.height // 2)
            self.port_radius = min(self.width, self.height) // 4
        
        self.port_size = int(self.port_radius * 4)
        
    def get_perspective_transform(self) -> np.ndarray:
        """Calculate perspective transform matrix."""
        half_size = self.port_size // 2
        src_points = np.float32([
            [self.port_center[0] - half_size, self.port_center[1] - half_size],
            [self.port_center[0] + half_size, self.port_center[1] - half_size],
            [self.port_center[0] + half_size, self.port_center[1] + half_size],
            [self.port_center[0] - half_size, self.port_center[1] + half_size]
        ])
        
        cos_angle = np.cos(self.angle_rad)
        sin_angle = np.sin(self.angle_rad)
        
        depth_scale = 0.8
        width_scale = cos_angle
        x_offset = half_size * sin_angle * 0.5
        
        dst_points = np.float32([
            [self.port_center[0] - half_size + x_offset, self.port_center[1] - half_size],
            [self.port_center[0] + half_size * width_scale + x_offset, self.port_center[1] - half_size],
            [self.port_center[0] + half_size * width_scale + x_offset, self.port_center[1] + half_size],
            [self.port_center[0] - half_size + x_offset, self.port_center[1] + half_size]
        ])
        
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        return M
    
    def apply_perspective(self) -> np.ndarray:
        """Apply perspective transformation to generate the view."""
        M = self.get_perspective_transform()
        
        perspective_view = cv2.warpPerspective(
            self.front_image,
            M,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return perspective_view
    
    def visualize(self):
        """Display the perspective view."""
        perspective_view = self.apply_perspective()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(perspective_view, cv2.COLOR_BGR2RGB))
        plt.title(f'View from Right Side\nDistance: {self.distance_cm}cm, Angle: {np.rad2deg(self.angle_rad):.1f}°')
        plt.axis('on')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return perspective_view

def main():
    """Generate a perspective view of the satellite port."""
    transformer = PerspectiveTransformer(distance_cm=100.0, angle_deg=22.5)
    
    print("\nGenerating perspective view...")
    print(f"Camera distance: 100 cm")
    print(f"Viewing angle: 22.5 degrees from right side")
    
    perspective_view = transformer.visualize()
    cv2.imwrite('perspective_view.png', perspective_view)
    print("\nPerspective view saved as 'perspective_view.png'")

if __name__ == "__main__":
    main() 