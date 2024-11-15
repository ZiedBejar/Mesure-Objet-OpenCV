import cv2
import numpy as np

# Known size of the QR code in millimeters
qr_code_size_mm = 34

# Camera calibration data (from your previous calibration results)
camera_matrix = np.array([[1.39250698e+03, 0.00000000e+00, 6.72249082e+02],
                          [0.00000000e+00, 1.40135905e+03, 3.60723059e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distortion_coefficients = np.array([1.10268789e-01, 7.26960791e-01, 1.33160063e-03, -2.27420497e-03, -3.77466718e+00])

# Focal length (F_x) from the camera matrix
focal_length = camera_matrix[0, 0]  # F_x value

# Start video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Show the frame in a window
    cv2.imshow('Video Feed (Press C to Capture)', frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # If the user presses 'c', capture the image and process it
    if key == ord('c'):
        print("Captured an image, starting processing...")

        # Undistort the captured frame using the camera calibration
        undistorted_img = cv2.undistort(frame, camera_matrix, distortion_coefficients)

        # Convert the undistorted image to grayscale
        gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

        # Detect QR codes
        detector = cv2.QRCodeDetector()
        data, bbox, rectified_image = detector.detectAndDecode(gray)

        if bbox is not None:
            print("QR Code detected!")
            print("Decoded data:", data)

            # Find the width and height of the QR code in pixels
            qr_code_width_pixels = np.linalg.norm(bbox[0][0] - bbox[0][1])

            # Calculate the pixel-to-mm conversion factor
            pixel_to_mm_ratio = qr_code_size_mm / qr_code_width_pixels

            # Calculate the distance from the camera to the QR code
            distance = ((focal_length * qr_code_size_mm) / qr_code_width_pixels)-320
            print(f"Estimated distance to the QR code: {distance:.2f} mm")

            # Apply Canny edge detection to find contours (for the black rectangle)
            edges = cv2.Canny(gray, 100, 200)

            # Find contours in the image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            black_rectangle_contour = None
            for contour in contours:
                # Approximate contour to a polygon
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Check if it has 4 vertices (rectangle)
                    black_rectangle_contour = approx
                    break

            if black_rectangle_contour is not None:
                # Calculate the bounding box of the rectangle
                x, y, w, h = cv2.boundingRect(black_rectangle_contour)

                # Convert the dimensions from pixels to millimeters
                w_mm = w * pixel_to_mm_ratio
                h_mm = h * pixel_to_mm_ratio

                print(f"Dimensions of the black rectangle: Width = {w_mm:.2f} mm, Height = {h_mm:.2f} mm")
            else:
                print("No black rectangle detected.")
        else:
            print("No QR code detected.")

    # If the user presses 'q', break the loop and exit
    elif key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
