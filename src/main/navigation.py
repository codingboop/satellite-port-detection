import numpy as np
import cv2
import matplotlib.pyplot as plt
from base_image import SatellitePort
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional

class CameraNavigator:
    def __init__(self, view_size: Tuple[int, int] = (200, 200)):
        """Initialize the camera navigator with the reference image from base_image.py."""
        # Get reference image from SatellitePort
        port = SatellitePort()
        self.full_image = port.generate_image()
            
        self.view_size = view_size
        self.current_pos = [0, 0]  # Top-left corner of current view
        self.search_positions = self._generate_spiral_search()
        self.search_index = 0
        self.found = False
        self.path_history = [self.current_pos.copy()]
        self.movement_history = []
        
    def _generate_spiral_search(self, step_size: int = 50) -> List[List[int]]:
        """Generate spiral pattern search positions."""
        positions = []
        x, y = self.full_image.shape[1]//2, self.full_image.shape[0]//2
        dx, dy = [0, -1, 0, 1], [-1, 0, 1, 0]  # Directions: left, down, right, up
        step_count = 1
        direction = 0
        steps_taken = 0
        
        while len(positions) < 100:  # Limit to prevent infinite loops
            x += dx[direction] * step_size
            y += dy[direction] * step_size
            positions.append([x - self.view_size[0]//2, y - self.view_size[1]//2])
            
            steps_taken += 1
            if steps_taken == step_count:
                steps_taken = 0
                direction = (direction + 1) % 4
                if direction % 2 == 0:
                    step_count += 1
                    
        return positions
        
    def get_current_view(self) -> np.ndarray:
        """Get the current camera view."""
        x, y = self.current_pos
        h, w = self.view_size
        
        # Ensure we don't go out of bounds
        x = max(0, min(x, self.full_image.shape[1] - w))
        y = max(0, min(y, self.full_image.shape[0] - h))
        
        # Note: OpenCV images are indexed as [y, x] since height comes first
        view = self.full_image[y:y+h, x:x+w].copy()
        
        # Draw a border around the view for better visibility
        cv2.rectangle(view, (0, 0), (w-1, h-1), (0, 255, 0), 2)
        
        return view
        
    def detect_circle(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Detect if any part of the red circle is visible in the current view."""
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (including both primary and secondary red hues)
        lower_red1 = np.array([0, 50, 50])  # More sensitive thresholds
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges and combine
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((3,3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for any significant red region
        min_area = 20  # Even lower threshold to detect smaller parts
        debug_image = image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Draw the contour for visualization
                cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 2)
                
                # Draw bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Add text showing the area
                cv2.putText(debug_image, f"Area: {area:.0f}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                return True, debug_image
                
        return False, debug_image
        
    def move_to_next_position(self) -> bool:
        """Move to the next search position."""
        if self.search_index >= len(self.search_positions):
            return False
            
        self.current_pos = self.search_positions[self.search_index].copy()
        self.search_index += 1
        self.path_history.append(self.current_pos.copy())
        return True
        
    def search_for_circle(self) -> Tuple[bool, List[str], Tuple[int, int]]:
        """Search for the circle and return movement commands and final relative position."""
        movements = []
        detection_frame = None
        total_dx, total_dy = 0, 0  # Track total movement
        
        while not self.found and self.move_to_next_position():
            current_view = self.get_current_view()
            self.found, debug_view = self.detect_circle(current_view)
            
            if not self.found:
                # Calculate movement from previous position
                prev_pos = self.path_history[-2]
                dx = self.current_pos[0] - prev_pos[0]
                dy = self.current_pos[1] - prev_pos[1]
                
                # Update total movement
                total_dx += dx
                total_dy += dy
                
                if dx > 0:
                    movements.append(f"Move right by {dx}px")
                elif dx < 0:
                    movements.append(f"Move left by {abs(dx)}px")
                    
                if dy > 0:
                    movements.append(f"Move down by {dy}px")
                elif dy < 0:
                    movements.append(f"Move up by {abs(dy)}px")
                
                # Store movement for animation
                self.movement_history.append({
                    'position': self.current_pos.copy(),
                    'view': debug_view.copy(),
                    'found': False,
                    'movement': movements[-1] if movements else "",
                    'total_movement': (total_dx, total_dy)
                })
            else:
                # Store final position where circle was found
                self.movement_history.append({
                    'position': self.current_pos.copy(),
                    'view': debug_view.copy(),
                    'found': True,
                    'movement': "Circle found!",
                    'total_movement': (total_dx, total_dy)
                })
                detection_frame = debug_view.copy()
            
        return self.found, movements, (total_dx, total_dy)

def animate_search(navigator: CameraNavigator):
    """Create an animated visualization of the search process."""
    fig = plt.figure(figsize=(15, 8))
    gs = plt.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    
    # Full image with search path
    ax_full = fig.add_subplot(gs[:, 0])
    ax_full.imshow(cv2.cvtColor(navigator.full_image, cv2.COLOR_BGR2RGB))
    ax_full.set_title('Full Image with Search Path')
    
    # Current camera view
    ax_view = fig.add_subplot(gs[0, 1])
    ax_view.set_title('Current Camera View')
    
    # Text area for movement commands
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off')
    
    # Initialize visualization elements
    search_line, = ax_full.plot([], [], 'b-', alpha=0.5, label='Search path')
    current_box = plt.Rectangle((0, 0), navigator.view_size[0], navigator.view_size[1],
                              fill=False, color='r', linewidth=2)
    ax_full.add_patch(current_box)
    
    # Initialize camera view with a blank image
    camera_view = ax_view.imshow(np.zeros((*navigator.view_size, 3), dtype=np.uint8))
    
    # Initialize text elements
    status_text = ax_text.text(0.1, 0.7, "", fontsize=10)
    movement_text = ax_text.text(0.1, 0.5, "", fontsize=10)
    position_text = ax_text.text(0.1, 0.3, "", fontsize=10)
    
    def init():
        search_line.set_data([], [])
        current_box.set_xy((0, 0))
        camera_view.set_array(np.zeros((*navigator.view_size, 3), dtype=np.uint8))
        status_text.set_text("")
        movement_text.set_text("")
        position_text.set_text("")
        return search_line, current_box, camera_view, status_text, movement_text, position_text
    
    def update(frame):
        if frame < len(navigator.movement_history):
            # Update search path
            path = np.array(navigator.path_history[:frame+1])
            search_line.set_data(path[:, 0] + navigator.view_size[0]//2,
                               path[:, 1] + navigator.view_size[1]//2)
            
            # Update current position box
            current_pos = navigator.movement_history[frame]['position']
            current_box.set_xy(current_pos)
            
            # Update camera view
            current_view = navigator.movement_history[frame]['view']
            camera_view.set_array(cv2.cvtColor(current_view, cv2.COLOR_BGR2RGB))
            
            # Update text
            movement = navigator.movement_history[frame]['movement']
            status = "Found!" if navigator.movement_history[frame]['found'] else "Searching..."
            total_dx, total_dy = navigator.movement_history[frame]['total_movement']
            
            status_text.set_text(f"Status: {status}")
            movement_text.set_text(f"Movement: {movement}")
            position_text.set_text(f"Relative Position: ({total_dx:+d}, {total_dy:+d})")
            
            if navigator.movement_history[frame]['found']:
                ax_view.set_title('Circle Detected!')
            else:
                ax_view.set_title('Current Camera View')
        
        return search_line, current_box, camera_view, status_text, movement_text, position_text
    
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(navigator.movement_history),
                        interval=1000, blit=True)
    
    plt.tight_layout()
    plt.show()

def main():
    # Create navigator
    navigator = CameraNavigator(view_size=(150, 150))  # Smaller view size for better visibility
    
    # Start from random position far from the center
    margin = 100  # Minimum distance from center
    center_x, center_y = navigator.full_image.shape[1]//2, navigator.full_image.shape[0]//2
    
    while True:
        start_x = np.random.randint(0, navigator.full_image.shape[1] - navigator.view_size[0])
        start_y = np.random.randint(0, navigator.full_image.shape[0] - navigator.view_size[1])
        
        # Check if position is far enough from center
        if abs(start_x - center_x) > margin or abs(start_y - center_y) > margin:
            break
    
    navigator.current_pos = [start_x, start_y]
    navigator.path_history = [navigator.current_pos.copy()]
    
    print(f"\nStarting search from position ({start_x}, {start_y})")
    print("This position is deliberately chosen to be far from the center.")
    print("Camera view size:", navigator.view_size)
    
    # Search for circle
    found, movements, (total_dx, total_dy) = navigator.search_for_circle()
    
    if found:
        print("\nCircle found! Movement commands:")
        for movement in movements:
            print(movement)
        print(f"\nFinal relative position from start: ({total_dx:+d}, {total_dy:+d})")
        print(f"Total distance moved: {abs(total_dx) + abs(total_dy)} pixels")
    else:
        print("\nCircle not found after exhaustive search")
    
    # Animate the search process
    animate_search(navigator)

if __name__ == "__main__":
    main() 