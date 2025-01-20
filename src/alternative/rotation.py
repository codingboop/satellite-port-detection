import numpy as np
import cv2
import matplotlib.pyplot as plt
from base_image import SatellitePort
import random

class RotationDetector:
    def __init__(self):
        self.port = SatellitePort(scale_factor=10)
        self.reference_image = self.port.generate_image()
        
    def rotate_image(self, image, angle):
        """Rotate image by given angle clockwise.
        OpenCV's rotation is counterclockwise, so we negate the angle."""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        return rotated_image
    
    def detect_circle(self, image):
        """Detect the red circle in the image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise Exception("Circle not detected!")
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            raise Exception("Could not calculate circle center!")
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy), red_mask
    
    def calculate_rotation_angle(self, original_center, rotated_center, image_center):
        """Calculate rotation angle based on circle position.
        Measures angle clockwise from original position to rotated position."""
        orig_rel = np.array([original_center[0] - image_center[0],
                           -(original_center[1] - image_center[1])])
        
        rot_rel = np.array([rotated_center[0] - image_center[0],
                          -(rotated_center[1] - image_center[1])])
        
        orig_angle = np.arctan2(orig_rel[1], orig_rel[0])
        rot_angle = np.arctan2(rot_rel[1], rot_rel[0])
        
        orig_angle_deg = np.degrees(orig_angle)
        rot_angle_deg = np.degrees(rot_angle)
        
        if orig_angle_deg < 0:
            orig_angle_deg += 360
        if rot_angle_deg < 0:
            rot_angle_deg += 360
            
        angle_diff = orig_angle_deg - rot_angle_deg
        if angle_diff < 0:
            angle_diff += 360
            
        return angle_diff, orig_rel, rot_rel
    
    def detect_rotation(self, rotated_image):
        """Detect rotation angle of the rotated image compared to reference."""
        height, width = rotated_image.shape[:2]
        image_center = (width // 2, height // 2)
        
        original_center, original_mask = self.detect_circle(self.reference_image)
        rotated_center, rotated_mask = self.detect_circle(rotated_image)
        
        angle, orig_vec, rot_vec = self.calculate_rotation_angle(original_center, rotated_center, image_center)
        return angle, original_center, rotated_center, image_center, original_mask, rotated_mask, orig_vec, rot_vec
    
    def visualize_detection_steps(self, original_image, rotated_image, detection_data, true_angle):
        """Create detailed visualization of the detection process."""
        angle, orig_center, rot_center, img_center, orig_mask, rot_mask, orig_vec, rot_vec = detection_data
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.plot([img_center[0], orig_center[0]], [img_center[1], orig_center[1]], 'g-', label='Original Vector')
        ax1.plot(orig_center[0], orig_center[1], 'go', label='Circle Center')
        ax1.plot(img_center[0], img_center[1], 'yo', label='Image Center')
        ax1.set_title('Original Image\nwith Detection Points')
        ax1.legend()
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
        ax2.plot([img_center[0], rot_center[0]], [img_center[1], rot_center[1]], 'r-', label='Rotated Vector')
        ax2.plot(rot_center[0], rot_center[1], 'ro', label='Circle Center')
        ax2.plot(img_center[0], img_center[1], 'yo', label='Image Center')
        ax2.set_title(f'Rotated Image\nTrue Angle: {true_angle:.2f}°\nDetected Angle: {angle:.2f}°')
        ax2.legend()
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(orig_mask, cmap='gray')
        ax3.set_title('Original Circle Detection Mask')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(rot_mask, cmap='gray')
        ax4.set_title('Rotated Circle Detection Mask')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot([0, orig_vec[0]], [0, orig_vec[1]], 'g-', label='Original Vector')
        ax5.plot([0, rot_vec[0]], [0, rot_vec[1]], 'r-', label='Rotated Vector')
        ax5.plot(0, 0, 'ko', label='Origin')
        ax5.set_title('Vector Comparison\nfor Angle Calculation')
        ax5.grid(True)
        ax5.legend()
        ax5.axis('equal')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.text(0.1, 0.9, 'Detection Process:', fontsize=12, fontweight='bold')
        ax6.text(0.1, 0.8, '1. Detect red circle in both images', fontsize=10)
        ax6.text(0.1, 0.7, '2. Find circle centers (colored dots)', fontsize=10)
        ax6.text(0.1, 0.6, '3. Create vectors from image center', fontsize=10)
        ax6.text(0.1, 0.5, '4. Calculate clockwise angle between vectors', fontsize=10)
        ax6.text(0.1, 0.3, f'True Angle: {true_angle:.2f}°', fontsize=12, fontweight='bold')
        ax6.text(0.1, 0.2, f'Detected Angle: {angle:.2f}°', fontsize=12, fontweight='bold')
        ax6.text(0.1, 0.1, f'Error: {abs(true_angle - angle):.2f}°', fontsize=12, fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        return fig
    
    def process_image(self):
        """Main process to rotate image and detect angle."""
        true_angle = random.uniform(0, 360)
        rotated_image = self.rotate_image(self.reference_image.copy(), true_angle)
        
        detection_data = self.detect_rotation(rotated_image)
        detected_angle = detection_data[0]
        
        fig = self.visualize_detection_steps(self.reference_image, rotated_image, detection_data, true_angle)
        plt.show()
        
        return true_angle, detected_angle

def main():
    detector = RotationDetector()
    true_angle, detected_angle = detector.process_image()
    print(f"\nRotation Analysis:")
    print(f"True Rotation Angle: {true_angle:.2f}°")
    print(f"Detected Rotation Angle: {detected_angle:.2f}°")
    print(f"Error: {abs(true_angle - detected_angle):.2f}°")

if __name__ == "__main__":
    main()