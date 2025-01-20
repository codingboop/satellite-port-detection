import numpy as np
import cv2
import matplotlib.pyplot as plt
from base_image import SatellitePort
import random

class EnhancedRotationDetector:
    def __init__(self):
        self.port = SatellitePort(scale_factor=10)
        self.reference_image = self.port.generate_image()
        
    def rotate_image(self, image, angle):
        # OpenCV's rotation is counterclockwise, so negate the angle
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        return rotated_image
    
    def detect_circle_advanced(self, image, is_reference=False):
        # Convert to HSV and isolate red
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red color masks
        lower_red1 = np.array([0, 30, 50])
        upper_red1 = np.array([20, 255, 255])
        lower_red2 = np.array([160, 30, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None, red_mask, 0.0, None
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 5:
            return None, red_mask, 0.0, None
        
        # Calculate true radius using geometric formula
        spacing_px = self.port.spacing_px
        true_radius = (spacing_px * np.sqrt(2)) / (1 + np.sqrt(2))
        
        if is_reference:
            # For reference image, use exact geometric center
            center_x = true_radius
            center_y = true_radius
            confidence = 1.0
        else:
            # For rotated image, use geometric constraints
            points = largest_contour.squeeze()
            
            # Calculate visibility ratio
            arc_length = cv2.arcLength(largest_contour, False)
            full_circumference = 2 * np.pi * true_radius
            visibility = arc_length / full_circumference
            
            # Find potential touching points using extremal points
            extremal_points = []
            
            # Get initial estimate from ellipse fit
            ellipse = cv2.fitEllipse(largest_contour)
            (init_x, init_y), (major_axis, minor_axis), angle = ellipse
            
            # Find points that could be touching the squares
            for point in points:
                is_near_edge = (point[0] < spacing_px/4 or 
                              point[0] > image.shape[1] - spacing_px/4 or
                              point[1] < spacing_px/4 or 
                              point[1] > image.shape[0] - spacing_px/4)
                
                if is_near_edge:
                    nearby_points = 0
                    for other_point in points:
                        if np.all(point != other_point):
                            dist = np.sqrt(np.sum((point - other_point)**2))
                            if dist < spacing_px/8:
                                nearby_points += 1
                    
                    if nearby_points >= 2:
                        extremal_points.append(point)
            
            if len(extremal_points) >= 2:
                # Find the two most distant points
                max_dist = 0
                p1, p2 = None, None
                for i, point1 in enumerate(extremal_points):
                    for point2 in extremal_points[i+1:]:
                        dist = np.sqrt(np.sum((point1 - point2)**2))
                        if dist > max_dist:
                            max_dist = dist
                            p1, p2 = point1, point2
                
                if p1 is not None and p2 is not None:
                    # Calculate center using geometric constraints
                    mid_point = (p1 + p2) / 2
                    vec = p2 - p1
                    perp_vec = np.array([-vec[1], vec[0]])
                    perp_vec = perp_vec / np.linalg.norm(perp_vec)
                    
                    half_chord = max_dist / 2
                    if true_radius > half_chord:
                        height = np.sqrt(true_radius**2 - half_chord**2)
                        
                        center1 = mid_point + height * perp_vec
                        center2 = mid_point - height * perp_vec
                        
                        best_center = None
                        min_error = float('inf')
                        
                        for center in [center1, center2]:
                            errors = []
                            for point in extremal_points:
                                dist = np.sqrt(np.sum((center - point)**2))
                                errors.append(abs(dist - true_radius))
                            avg_error = np.mean(errors)
                            
                            if avg_error < min_error:
                                min_error = avg_error
                                best_center = center
                        
                        if min_error < spacing_px/4:
                            center_x, center_y = best_center
                        else:
                            center_x, center_y = init_x, init_y
                    else:
                        center_x, center_y = init_x, init_y
                else:
                    center_x, center_y = init_x, init_y
            else:
                center_x, center_y = init_x, init_y
            
            # Calculate confidence based on geometric fit
            distances = []
            for point in points:
                dist = np.sqrt((point[0] - center_x)**2 + (point[1] - center_y)**2)
                distances.append(abs(dist - true_radius))
            avg_error = np.mean(distances)
            confidence = 1.0 / (1.0 + avg_error/true_radius)
            confidence = confidence * visibility

        # Create debug visualization
        debug_image = image.copy()
        
        # Draw grid and detected features
        for i in range(0, image.shape[1], spacing_px//4):
            cv2.line(debug_image, (i, 0), (i, image.shape[0]), (200, 200, 200), 1)
        for i in range(0, image.shape[0], spacing_px//4):
            cv2.line(debug_image, (0, i), (image.shape[1], i), (200, 200, 200), 1)
        
        cv2.drawContours(debug_image, [largest_contour], -1, (0, 255, 0), 2)
        
        if not is_reference and 'extremal_points' in locals():
            for point in extremal_points:
                cv2.circle(debug_image, (int(point[0]), int(point[1])), 3, (255, 0, 255), -1)
            
            if 'p1' in locals() and p1 is not None and p2 is not None:
                cv2.line(debug_image, 
                        (int(p1[0]), int(p1[1])), 
                        (int(p2[0]), int(p2[1])), 
                        (0, 255, 255), 1)
        
        cv2.circle(debug_image, (int(center_x), int(center_y)), int(true_radius), (0, 0, 255), 2)
        cv2.circle(debug_image, (int(center_x), int(center_y)), 3, (255, 255, 0), -1)
        
        cv2.putText(debug_image,
                   f"Confidence: {confidence:.2%}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 0, 0),
                   2)
        
        return (center_x, center_y), red_mask, confidence, debug_image
        
    def calculate_rotation_angle(self, original_center, rotated_center, image_center):
        # Get vectors from image center to circle centers
        orig_vec = np.array([original_center[0] - image_center[0],
                           original_center[1] - image_center[1]])
        
        rot_vec = np.array([rotated_center[0] - image_center[0],
                           rotated_center[1] - image_center[1]])
        
        # Normalize vectors and calculate angle using atan2
        orig_vec = orig_vec / np.linalg.norm(orig_vec)
        rot_vec = rot_vec / np.linalg.norm(rot_vec)
        
        angle_deg = np.degrees(np.arctan2(rot_vec[1], rot_vec[0]) - 
                             np.arctan2(orig_vec[1], orig_vec[0]))
        
        angle_deg = angle_deg % 360
        
        return angle_deg, orig_vec, rot_vec
        
    def process_image(self, fixed_angle=None):
        ref_center, ref_mask, ref_conf, ref_debug = self.detect_circle_advanced(self.reference_image, is_reference=True)
        
        if ref_center is None:
            raise Exception("Could not detect circle in reference image")
            
        if fixed_angle is not None:
            true_angle = fixed_angle
        else:
            true_angle = random.uniform(0, 360)
            
        rotated_image = self.rotate_image(self.reference_image, true_angle)
        rotated_center, rot_mask, rot_conf, rot_debug = self.detect_circle_advanced(rotated_image)
        
        if rotated_center is None:
            raise Exception("Could not detect circle in rotated image")
            
        image_center = (rotated_image.shape[1] // 2, rotated_image.shape[0] // 2)
        detected_angle, orig_vec, rot_vec = self.calculate_rotation_angle(ref_center, rotated_center, image_center)
        
        error = abs(detected_angle - true_angle)
        error = min(error, 360 - error)
        
        # Visualize results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
        
        ax1.imshow(cv2.cvtColor(ref_debug, cv2.COLOR_BGR2RGB))
        ax1.set_title('Reference Image Detection')
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(rot_debug, cv2.COLOR_BGR2RGB))
        ax2.set_title('Rotated Image Detection')
        ax2.axis('off')
        
        ax3.imshow(ref_mask, cmap='gray')
        ax3.set_title('Reference Mask')
        ax3.axis('off')
        
        ax4.imshow(rot_mask, cmap='gray')
        ax4.set_title('Rotated Mask')
        ax4.axis('off')
        
        plt.suptitle(f'Rotation Analysis\nTrue: {true_angle:.2f}°, Detected: {detected_angle:.2f}°\n'
                    f'Error: {error:.2f}°, Confidence: {rot_conf:.2%}')
        plt.tight_layout()
        plt.show()
        
        return true_angle, detected_angle, error, rot_conf

def main():
    detector = EnhancedRotationDetector()
    
    while True:
        try:
            angle_input = input("Enter rotation angle (0-360 degrees) or 'q' to quit: ")
            if angle_input.lower() == 'q':
                break
                
            angle = float(angle_input)
            if not 0 <= angle <= 360:
                print("Invalid angle. Please enter a value between 0 and 360.")
                continue
                
            true_angle, detected_angle, error, confidence = detector.process_image(angle)
            print(f"\nRotation Analysis:")
            print(f"True Angle: {true_angle:.2f}°")
            print(f"Detected Angle: {detected_angle:.2f}°")
            print(f"Error: {error:.2f}°")
            print(f"Detection Confidence: {confidence:.2%}\n")
            
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 360.")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 