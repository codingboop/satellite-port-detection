import numpy as np
import cv2
import matplotlib.pyplot as plt
from base_image import SatellitePort
from perspective_view import PerspectiveTransformer
import time

class CameraMovement:
    def __init__(self, start_angle_deg: float = 22.5, distance_cm: float = 100.0, steps: int = 10):
        """Initialize camera movement parameters."""
        self.distance_cm = distance_cm
        self.start_angle = start_angle_deg
        self.steps = steps
        
        self.transformer = PerspectiveTransformer(distance_cm=distance_cm, angle_deg=start_angle_deg)
        self.angles = np.linspace(start_angle_deg, 0, steps)
        
        self.positions = []
        for angle in self.angles:
            angle_rad = np.deg2rad(angle)
            x = distance_cm * np.sin(angle_rad)
            z = distance_cm * np.cos(angle_rad)
            self.positions.append((x, z))
    
    def generate_view(self, angle_deg: float) -> np.ndarray:
        """Generate view from specified angle."""
        self.transformer = PerspectiveTransformer(distance_cm=self.distance_cm, angle_deg=angle_deg)
        return self.transformer.apply_perspective()
    
    def animate_movement(self):
        """Animate camera movement from side view to front view."""
        plt.figure(figsize=(15, 8))
        
        gs = plt.GridSpec(1, 2, width_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        
        ax2.plot([0], [0], 'ro', label='Port')
        trajectory_x = [pos[0] for pos in self.positions]
        trajectory_z = [pos[1] for pos in self.positions]
        ax2.plot(trajectory_x, trajectory_z, 'b-', label='Camera Path')
        ax2.set_xlabel('X Position (cm)')
        ax2.set_ylabel('Z Position (cm)')
        ax2.set_title('Camera Trajectory')
        ax2.grid(True)
        ax2.legend()
        ax2.axis('equal')
        
        for i, (angle, position) in enumerate(zip(self.angles, self.positions)):
            ax1.clear()
            
            current_view = self.generate_view(angle)
            ax1.imshow(cv2.cvtColor(current_view, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'Camera View\nDistance: {self.distance_cm}cm, Angle: {angle:.1f}°\nStep {i+1}/{self.steps}')
            ax1.axis('on')
            ax1.grid(True)
            
            ax2.plot(position[0], position[1], 'bo')
            
            plt.tight_layout()
            plt.pause(0.5)
            
            cv2.imwrite(f'camera_view_step_{i+1}.png', current_view)
            print(f"\nSaved view at angle {angle:.1f}° (position: x={position[0]:.1f}cm, z={position[1]:.1f}cm)")
        
        plt.show()

def main():
    camera = CameraMovement(start_angle_deg=22.5, distance_cm=100.0, steps=10)
    
    print("\nStarting camera movement simulation...")
    print(f"Distance from port: {camera.distance_cm} cm")
    print(f"Moving from {camera.start_angle}° to 0° in {camera.steps} steps")
    print("Maintaining constant distance of 100cm from port center")
    
    camera.animate_movement()
    print("\nCamera movement simulation completed.")

if __name__ == "__main__":
    main() 