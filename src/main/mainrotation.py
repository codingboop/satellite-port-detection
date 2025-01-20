import numpy as np
import cv2
import matplotlib.pyplot as plt
from base_image import SatellitePort
import random

class EnhancedRotationDetector:
    def __init__(self):
        # Generate reference image
        self.port = SatellitePort(scale_factor=10)
        self.reference_image = self.port.generate_image()
        
    def rotate_image(self, image, angle):
        """
        Rotate image by given angle clockwise.
        OpenCV's rotation is counterclockwise, so we negate the angle.
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        return rotated_image
    
    def detect_circle_advanced(self, image, is_reference=False):
        """
        Detect circle and find its center using strict geometric constraints.
        For both reference and rotated images:
        - Radius = (spacing_px * sqrt(2)) / (1 + sqrt(2))
        - Center must maintain equal distances from touching points
        """
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

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            print("No contours found")
            return None, red_mask, 0.0, None
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 5:
            print("Contour too small")
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
            
            # Find potential touching points by looking at extremal points
            extremal_points = []
            
            # Get initial estimate from ellipse fit
            ellipse = cv2.fitEllipse(largest_contour)
            (init_x, init_y), (major_axis, minor_axis), angle = ellipse
            
            # Find points that could be touching the squares
            for point in points:
                # Check if point is near edges or corners
                is_near_edge = (point[0] < spacing_px/4 or 
                              point[0] > image.shape[1] - spacing_px/4 or
                              point[1] < spacing_px/4 or 
                              point[1] > image.shape[0] - spacing_px/4)
                
                if is_near_edge:
                    # Verify it's a potential touching point by checking nearby points
                    nearby_points = 0
                    for other_point in points:
                        if np.all(point != other_point):
                            dist = np.sqrt(np.sum((point - other_point)**2))
                            if dist < spacing_px/8:  # Points within small radius
                                nearby_points += 1
                    
                    if nearby_points >= 2:  # At least 2 other points nearby
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
                    # These points should be on opposite sides of the circle
                    # Their midpoint gives us one constraint on the center
                    mid_point = (p1 + p2) / 2
                    
                    # The center must be true_radius away from both points
                    # This gives us two circles - their intersection is our center
                    # We can find this by going perpendicular to the line p1->p2
                    vec = p2 - p1
                    perp_vec = np.array([-vec[1], vec[0]])
                    perp_vec = perp_vec / np.linalg.norm(perp_vec)
                    
                    # Calculate the height of the triangle formed by p1, p2, and center
                    # using the Pythagorean theorem
                    half_chord = max_dist / 2
                    if true_radius > half_chord:  # Ensure we don't get NaN
                        height = np.sqrt(true_radius**2 - half_chord**2)
                        
                        # Try both possible center positions
                        center1 = mid_point + height * perp_vec
                        center2 = mid_point - height * perp_vec
                        
                        # Choose the center that better matches our geometric constraints
                        best_center = None
                        min_error = float('inf')
                        
                        for center in [center1, center2]:
                            # Check distances to all extremal points
                            errors = []
                            for point in extremal_points:
                                dist = np.sqrt(np.sum((center - point)**2))
                                errors.append(abs(dist - true_radius))
                            avg_error = np.mean(errors)
                            
                            if avg_error < min_error:
                                min_error = avg_error
                                best_center = center
                        
                        if min_error < spacing_px/4:  # If error is reasonable
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
        
        # Draw grid lines
        for i in range(0, image.shape[1], spacing_px//4):
            cv2.line(debug_image, (i, 0), (i, image.shape[0]), (200, 200, 200), 1)
        for i in range(0, image.shape[0], spacing_px//4):
            cv2.line(debug_image, (0, i), (image.shape[1], i), (200, 200, 200), 1)
        
        # Draw detected contour
        cv2.drawContours(debug_image, [largest_contour], -1, (0, 255, 0), 2)
        
        # Draw extremal points if available
        if not is_reference and 'extremal_points' in locals():
            for point in extremal_points:
                cv2.circle(debug_image, (int(point[0]), int(point[1])), 3, (255, 0, 255), -1)
            
            # Draw the chord between most distant points if found
            if 'p1' in locals() and p1 is not None and p2 is not None:
                cv2.line(debug_image, 
                        (int(p1[0]), int(p1[1])), 
                        (int(p2[0]), int(p2[1])), 
                        (0, 255, 255), 1)
        
        # Draw reconstructed circle
        cv2.circle(debug_image, (int(center_x), int(center_y)), int(true_radius), (0, 0, 255), 2)
        
        # Draw center point
        cv2.circle(debug_image, (int(center_x), int(center_y)), 3, (255, 255, 0), -1)
        
        # Add confidence annotation
        cv2.putText(debug_image,
                   f"Confidence: {confidence:.2%}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   (0, 0, 0),
                   2)
        
        return (center_x, center_y), red_mask, confidence, debug_image
        
    def calculate_rotation_angle(self, original_center, rotated_center, image_center):
        """Calculate rotation angle between original and rotated circle centers."""
        # Get vectors from image center to circle centers
        orig_vec = np.array([original_center[0] - image_center[0],
                           original_center[1] - image_center[1]])
        
        rot_vec = np.array([rotated_center[0] - image_center[0],
                           rotated_center[1] - image_center[1]])
        
        # Normalize vectors to unit length
        orig_vec = orig_vec / np.linalg.norm(orig_vec)
        rot_vec = rot_vec / np.linalg.norm(rot_vec)
        
        # Calculate angle using atan2 for better handling of all quadrants
        angle_deg = np.degrees(np.arctan2(rot_vec[1], rot_vec[0]) - 
                             np.arctan2(orig_vec[1], orig_vec[0]))
        
        # Ensure angle is in [0, 360) range
        angle_deg = angle_deg % 360
        
        # For visualization, also return the original vectors
        return angle_deg, orig_vec, rot_vec
        
    def process_image(self, fixed_angle=None):
        """Process image with given or random rotation angle."""
        # Get reference circle center
        ref_center, ref_mask, ref_conf, ref_debug = self.detect_circle_advanced(self.reference_image, is_reference=True)
        
        if ref_center is None:
            raise Exception("Could not detect circle in reference image")
            
        # Generate rotated image
        if fixed_angle is not None:
            true_angle = fixed_angle
        else:
            true_angle = random.uniform(0, 360)
            
        rotated_image = self.rotate_image(self.reference_image, true_angle)
        
        # Detect circle in rotated image
        rotated_center, rotated_mask, rot_conf, rot_debug = self.detect_circle_advanced(rotated_image)
        
        if rotated_center is None:
            raise Exception("Could not detect circle in rotated image")
            
        # Calculate rotation angle
        image_center = (rotated_image.shape[1] // 2, rotated_image.shape[0] // 2)
        detected_angle, orig_vec, rot_vec = self.calculate_rotation_angle(ref_center, rotated_center, image_center)
        
        # Create visualization with enhanced angle display
        fig = plt.figure(figsize=(15, 10))
        
        # Original image with detection points
        ax1 = plt.subplot(231)
        ax1.imshow(cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB))
        ax1.plot([image_center[0], ref_center[0]], [image_center[1], ref_center[1]], 'g-', label='Original Vector')
        ax1.plot(ref_center[0], ref_center[1], 'go', label='Circle Center')
        ax1.plot(image_center[0], image_center[1], 'yo', label='Image Center')
        ax1.set_title('Original Image\nwith Detection Points')
        ax1.legend()
        
        # Rotated image with detection points and reconstructed circle
        ax2 = plt.subplot(232)
        ax2.imshow(cv2.cvtColor(rot_debug, cv2.COLOR_BGR2RGB))  # Use debug image with reconstruction
        ax2.plot([image_center[0], rotated_center[0]], [image_center[1], rotated_center[1]], 'r-', label='Rotated Vector')
        ax2.plot(rotated_center[0], rotated_center[1], 'ro', label='Circle Center')
        ax2.plot(image_center[0], image_center[1], 'yo', label='Image Center')
        ax2.set_title(f'Rotated Image with Reconstruction\nTrue Angle: {true_angle:.2f}°\nDetected Angle: {detected_angle:.2f}°')
        ax2.legend()
        
        # Original circle detection mask
        ax3 = plt.subplot(233)
        ax3.imshow(ref_mask, cmap='gray')
        ax3.set_title('Original Circle Detection Mask')
        
        # Rotated circle detection mask
        ax4 = plt.subplot(234)
        ax4.imshow(rotated_mask, cmap='gray')
        ax4.set_title('Rotated Circle Detection Mask')
        ax4.text(10, 30, f'Confidence: {rot_conf:.2%}', color='black')
        
        # Vector comparison with angle arc
        ax5 = plt.subplot(235)
        # Draw vectors
        ax5.plot([0, orig_vec[0]], [0, orig_vec[1]], 'g-', label='Original Vector')
        ax5.plot([0, rot_vec[0]], [0, rot_vec[1]], 'r-', label='Rotated Vector')
        ax5.plot(0, 0, 'ko', label='Origin')
        
        # Draw angle arc
        radius = 0.3
        angle_rad = np.radians(detected_angle)
        theta = np.linspace(0, angle_rad, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax5.plot(x, y, 'b--', alpha=0.5)
        
        # Add angle label at arc midpoint
        mid_angle = angle_rad / 2
        label_x = radius * 1.2 * np.cos(mid_angle)
        label_y = radius * 1.2 * np.sin(mid_angle)
        ax5.text(label_x, label_y, f'{detected_angle:.1f}°', ha='center', va='center')
        
        ax5.grid(True)
        ax5.axis('equal')
        ax5.set_title('Vector Comparison\nfor Angle Calculation')
        ax5.legend()
        
        # Detection process steps
        ax6 = plt.subplot(236)
        steps = [
            "1. Detect partial circle",
            "2. Reconstruct full circle geometry",
            "3. Find true center (even if outside)",
            "4. Calculate rotation angle using atan2"
        ]
        ax6.text(0.1, 0.8, '\n'.join(steps), transform=ax6.transAxes)
        ax6.text(0.1, 0.3, f'True Angle: {true_angle:.2f}°\nDetected Angle: {detected_angle:.2f}°\nError: {abs(true_angle - detected_angle):.2f}°\nVisibility: {rot_conf:.2%}', transform=ax6.transAxes)
        ax6.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return true_angle, detected_angle, rot_conf

def main():
    detector = EnhancedRotationDetector()
    
    while True:
        try:
            # Get angle input from user with improved prompt
            angle_str = input("\nEnter rotation angle (0-360 degrees) or 'q' to quit: ").strip().lower()
            
            # Check for quit command
            if angle_str == 'q':
                print("Exiting program...")
                break
                
            # Validate numeric input
            try:
                angle = float(angle_str)
                if not (0 <= angle <= 360):
                    print("Error: Angle must be between 0 and 360 degrees")
                    continue
            except ValueError:
                print("Error: Please enter a valid number or 'q' to quit")
                continue
                
            print(f"\nProcessing image with {angle:.1f}° rotation...")
            
            try:
                # Process image with user's angle
                true_angle, detected_angle, confidence = detector.process_image(fixed_angle=angle)
                
                # Print results with improved formatting
                print("\nRotation Analysis:")
                print("─" * 40)
                print(f"True Rotation Angle:    {true_angle:.2f}°")
                print(f"Detected Angle:         {detected_angle:.2f}°")
                print(f"Error:                  {abs(true_angle - detected_angle):.2f}°")
                print(f"Detection Confidence:   {confidence:.2%}")
                print("─" * 40)
                
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                print("Please try a different angle")
                continue
            
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("Please try again")
            continue

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        print("Program terminated") 