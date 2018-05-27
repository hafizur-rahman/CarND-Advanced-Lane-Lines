import cv2
import numpy as np

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)

    # Return this mask as your binary_output image
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return sxbinary


def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):    
    # Take the gradient in x and y separately
    sobelx= cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely= cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the magnitude 
    grad_mag = np.sqrt(sobelx**2 + sobely**2)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(grad_mag)/255
    scaled_sobel = (grad_mag/scale_factor).astype(np.uint8) 
    
    # Create a binary mask where mag thresholds are met
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return this mask as your binary_output image
    return sbinary


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):    
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad_dir)
    
    # return this mask as your binary_output image
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return binary_output

def region_of_interest(img, vertices):
    # Defining a blank mask to start with
    mask = np.zeros_like(img)   

    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100), R_thresh=(5, 255), sobel_kernel = 3, blur=True):    
    kernel_size = 5
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Pull R
    R = img[:,:,0]
    
    # Threshold R color channel
    R_binary = np.zeros_like(R)
    R_binary[(R > R_thresh[0]) & (R <= R_thresh[1])] = 1
    
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel > sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Combine the binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[((R_binary == 1)) & ((sx_binary == 1) | (s_binary == 1))] = 1
    
    # Define vertices for marked area
    # left_b = (100, img.shape[0])
    # right_b = (img.shape[1]-20, img.shape[0])
    # apex1 = (610, 410)
    # apex2 = (680, 410)
    # inner_left_b = (310, img.shape[0])
    # inner_right_b = (1150, img.shape[0])
    # inner_apex1 = (700,480)
    # inner_apex2 = (650,480)
    # vertices = np.array([[left_b, apex1, apex2, right_b, inner_right_b, \
    #                       inner_apex1, inner_apex2, inner_left_b]], dtype=np.int32)

    # # Select region of interest
    # combined_binary = region_of_interest(combined_binary, vertices)
    
    return combined_binary

def birds_eye_view(image, mtx, dist):  
    # Undistort the image using calibration params found earlier
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    
    # Do perspective transform
    # Grab the image shape
    img_size = (image.shape[1], image.shape[0])   
    
    # Source points - defined area of lane line edges
    src = np.float32([[690,450],[1110,img_size[1]],[175,img_size[1]],[595,450]])

    # 4 destination points to transfer
    offset = 300 # offset for dst points
    
    dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]],[offset, 0]])
    
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # use cv2.warpPerspective() to warp your image to a top-down view
    unwarped = cv2.warpPerspective(undistorted, M, img_size)        
    
    return unwarped, M, Minv, src, dst