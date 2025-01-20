import numpy as np
import cv2
import matplotlib.pyplot as plt
from base_image import SatellitePort
from perspective_view import PerspectiveTransformer

class UnifiedVisualizer:
    def __init__(self, distance_cm: float = 100.0):
        """Initialize unified visualization system.
        Args:
            distance_cm: Distance from camera to port center (default: 100.0 cm)
        """
        self.distance_cm = distance_cm
        self.port = SatellitePort()
        self.reference_image = self.port.generate_image()
        
    def generate_perspective_view(self, angle_deg: float) -> np.ndarray:
        """Generate perspective view for a given angle."""
        if not 0 <= angle_deg <= 90:
            raise ValueError("Angle must be between 0 and 90 degrees")
            
        transformer = PerspectiveTransformer(distance_cm=self.distance_cm, angle_deg=angle_deg)
        return transformer.apply_perspective()
    
    def calculate_camera_positions(self, start_angle: float, steps: int) -> list:
        """Calculate camera positions for movement animation."""
        angles = np.linspace(start_angle, 0, steps)
        positions = []
        for angle in angles:
            angle_rad = np.deg2rad(angle)
            x = self.distance_cm * np.sin(angle_rad)
            z = self.distance_cm * np.cos(angle_rad)
            positions.append((x, z, angle))
        return positions
    
    def visualize_unified(self, angle_deg: float, animate: bool = False, steps: int = 20):
        """Create unified visualization with static view and optional animation."""
        plt.figure(figsize=(20, 10))
        
        # Create 2x2 grid of subplots
        gs = plt.GridSpec(2, 2)
        
        # Top left: Original reference image
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Reference Image (Top View)')
        ax1.axis('on')
        ax1.grid(True)
        
        # Top right: Static perspective view
        ax2 = plt.subplot(gs[0, 1])
        perspective_view = self.generate_perspective_view(angle_deg)
        ax2.imshow(cv2.cvtColor(perspective_view, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Perspective View\nAngle: {angle_deg}°, Distance: {self.distance_cm}cm')
        ax2.axis('on')
        ax2.grid(True)
        
        # Bottom left: Camera trajectory
        ax3 = plt.subplot(gs[1, 0])
        ax3.plot([0], [0], 'ro', markersize=10, label='Port')  # Port at origin
        
        if animate:
            positions = self.calculate_camera_positions(angle_deg, steps)
            
            # Draw the full circular arc for reference
            arc_angles = np.linspace(0, angle_deg, 100)
            arc_x = self.distance_cm * np.sin(np.deg2rad(arc_angles))
            arc_z = self.distance_cm * np.cos(np.deg2rad(arc_angles))
            ax3.plot(arc_x, arc_z, 'k--', alpha=0.3, label='Full Path')
            
            # Initialize empty line for animated trajectory
            trajectory_line, = ax3.plot([], [], 'b-', linewidth=2, label='Camera Path')
            current_pos, = ax3.plot([], [], 'gs', markersize=10, label='Camera Position')
            
            # Set fixed limits for trajectory plot
            ax3.set_xlim(-self.distance_cm * 1.1, self.distance_cm * 1.1)
            ax3.set_ylim(-self.distance_cm * 0.1, self.distance_cm * 1.1)
            
            # Bottom right: Animation panel
            ax4 = plt.subplot(gs[1, 1])
            
            # Initialize lists for animated trajectory
            traj_x, traj_z = [], []
            
            # Animation loop
            for i, (x, z, angle) in enumerate(positions):
                # Update trajectory
                traj_x.append(x)
                traj_z.append(z)
                trajectory_line.set_data(traj_x, traj_z)
                current_pos.set_data([x], [z])
                
                # Update perspective view
                ax4.clear()
                view = self.generate_perspective_view(angle)
                ax4.imshow(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
                ax4.set_title(f'Animated View\nAngle: {angle:.1f}°\nPosition: x={x:.1f}cm, z={z:.1f}cm')
                ax4.axis('on')
                ax4.grid(True)
                
                # Add direction arrow
                arrow_length = self.distance_cm * 0.2
                dx = -arrow_length * np.sin(np.deg2rad(angle))
                dz = -arrow_length * np.cos(np.deg2rad(angle))
                ax3.arrow(x, z, dx, dz, head_width=3, head_length=5, fc='g', ec='g', alpha=0.5)
                
                # Update display
                plt.pause(0.1)
                
                # Save frame
                cv2.imwrite(f'camera_view_angle_{angle:.1f}.png', view)
                
                # Update title with progress
                ax3.set_title(f'Camera Trajectory\nProgress: {(i+1)}/{len(positions)} steps')
        else:
            # Draw static trajectory
            start_x = self.distance_cm * np.sin(np.deg2rad(angle_deg))
            start_z = self.distance_cm * np.cos(np.deg2rad(angle_deg))
            ax3.plot(start_x, start_z, 'gs', markersize=10, label='Camera Position')
            
            # Add direction arrow
            arrow_length = self.distance_cm * 0.2
            dx = -arrow_length * np.sin(np.deg2rad(angle_deg))
            dz = -arrow_length * np.cos(np.deg2rad(angle_deg))
            ax3.arrow(start_x, start_z, dx, dz, head_width=3, head_length=5, fc='g', ec='g', alpha=0.5)
            
            # Add message to animation panel
            ax4 = plt.subplot(gs[1, 1])
            ax4.text(0.5, 0.5, 'Select "y" for animation\nto see camera movement', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Animation Panel')
        
        ax3.set_xlabel('X Position (cm)')
        ax3.set_ylabel('Z Position (cm)')
        ax3.grid(True)
        ax3.legend(loc='upper right')
        ax3.axis('equal')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create unified visualizer
    visualizer = UnifiedVisualizer(distance_cm=100.0)
    
    while True:
        try:
            # Get angle input
            angle_input = input("\nEnter viewing angle (0-90 degrees) or 'q' to quit: ")
            if angle_input.lower() == 'q':
                break
                
            angle = float(angle_input)
            if not 0 <= angle <= 90:
                print("Angle must be between 0 and 90 degrees")
                continue
            
            # Get animation preference
            animate_input = input("Animate camera movement? (y/n): ")
            animate = animate_input.lower() == 'y'
            
            # Create visualization
            print(f"\nGenerating visualization for {angle}° viewing angle...")
            visualizer.visualize_unified(angle, animate=animate, steps=20)
            
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            print("\nVisualization interrupted")
            break
    
    print("\nVisualization completed.")

if __name__ == "__main__":
    main() 