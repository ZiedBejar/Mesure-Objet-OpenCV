import cv2
import numpy as np

# Load the image
img = cv2.imread('C:\\Users\\LENOVO\\Pictures\\Camera Roll\\WIN_20241010_00_07_31_Pro.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect QR codes
detector = cv2.QRCodeDetector()
data, bbox, rectified_image = detector.detectAndDecode(gray)

# Print the decoded data
print("Decoded data:", data)

# Apply canny edge detection
canny = cv2.Canny(gray, 100, 200)

# Find contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour of the rectangle
rect_contour = None
for contour in contours:
    # Check if the contour is a rectangle
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        rect_contour = approx
        break

# If the rectangle is found
if rect_contour is not None:
    # Find the dimensions of the rectangle
    x, y, w, h = cv2.boundingRect(rect_contour)
    print("Dimensions of the black rectangle:", w, h)

    # Draw the rectangle on the original image for visualization
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    print("No rectangle found.")

# Get the camera calibration matrix and distortion coefficients
camera_matrix = np.array([[900.63984412, 0, 620.72593803],
                          [0, 903.01829889, 355.53299236],
                          [0, 0, 1]])
distortion_coefficients = np.array([[-0.21012947, 0.64500762, -0.00128072, -0.0046403, -0.80714926]])

# Undistort the original image
undistorted_image = cv2.undistort(img, camera_matrix, distortion_coefficients)

# Calculate the known width of the QR code in cm (assumed here, replace with actual size)
known_width_cm = 4  # Example: QR code width in cm
pixel_width = w  # Width of detected rectangle in pixels

# Calculate the distance from the camera using the focal length and known width
focal_length = camera_matrix[0, 0]  # fx from the camera matrix
distance = (known_width_cm * focal_length) / pixel_width

print("Distance of the QR code from the camera:", distance)

# Show images for visualization
cv2.imshow("Original Image", img)
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
