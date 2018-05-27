import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

def calibrate_camera(calib_images_path, annotate=False):
    '''
    If `annotate` is True, annotated images are returned as well
    '''
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    annotated_images = []

    # Step through the list and search for chessboard corners
    for fname in calib_images_path:
        img = mpimage.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        found, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if found == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if annotate:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, found)                
                annotated_images.append(img)
    
    # Calibrate the camera
    _, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)
    
    return mtx, dist, annotated_images


def undistort(distorted_image, mtx, dist):
    return cv2.undistort(distorted_image, mtx, dist, None, mtx)