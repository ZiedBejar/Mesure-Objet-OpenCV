import numpy as np
import cv2 as cv
import glob

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (3D points in real-world space)
chessboard_rows = 4  # Number of inner rows
chessboard_cols = 6   # Number of inner columns
square_size = 2.5     # Size of a square in cm

# Prepare object points based on the size of the squares
objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load images for calibration
images = glob.glob('C:\\Users\\LENOVO\\Downloads\\image2\\*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (chessboard_cols, chessboard_rows), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print(f"Chessboard not found in {fname}")

cv.destroyAllWindows()

# Perform camera calibration
if len(objpoints) > 0 and len(imgpoints) > 0:
    if len(objpoints) == len(imgpoints):
        # Get the size of the last captured image
        h, w = gray.shape[:2]

        # Perform calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

        # Check if calibration was successful
        if ret:
            print("Camera calibration successful.")
            print("Camera matrix:\n", mtx)
            print("Distortion coefficients:\n", dist)
            print("Rotation vectors:\n", rvecs)
            print("Translation vectors:\n", tvecs)
        else:
            print("Camera calibration failed.")
    else:
        print(f"Mismatch: {len(objpoints)} object points and {len(imgpoints)} image points.")
else:
    print("No points collected for calibration.")
