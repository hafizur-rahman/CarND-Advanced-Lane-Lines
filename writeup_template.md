**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[output1]: ./output_images/undistort_output2.jpg "Undistorted Test Image"
[image3]: ./output_images/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[image7]: ./output_images/histogram.jpg "Histogram"
[image8]: ./output_images/sliding_window.jpg "Sliding Window"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in a function named `calibrate_camera()` at `calibrate.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, distortion correction was applied to one of the test images:
![alt text][output1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at `pipeline()` method lines #81 through #128 in `transform.py`).

Based on lecture videos, color threshold was done on R channel (in RGB color space) and S channel (in HLS color space). Gradient threshold was applied only in x direction.

To further reduce noise, a ROI (region of interest) mask was applied as well.

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `birds_eye_view()`, which appears in lines 130 through 154 in the file `transform.py`.  The `birds_eye_view()` function takes as inputs an image (`img`), as well as camera calibration related params like `mtx` and `dist`.

First, image is undisorted using camera calibration params. Then perspective transform is done using cv2 library. I chose the hardcode the source and destination points in the following manner:

```python
    img_size = (image.shape[1], image.shape[0])   
    
    # Source points - defined area of lane line edges
    src = np.float32([[690,450],[1110,img_size[1]],[175,img_size[1]],[595,450]])

    # 4 destination points to transfer
    offset = 300 # offset for dst points
    
    dst = np.float32([[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]],[offset, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 690, 450      | 980, 0        | 
| 1110, 720     | 980, 720      |
| 175, 720      | 300, 720      |
| 595, 450      | 300, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for detecting lane resides at `lane.py`.

For the first frame in the video, sliding window search was done (`get_lanes_sliding(binary_warped)`).

Histogram from bottom part of the binary unwarped image was created and then lane start was identified as highest frequency bins from left half and right half.

![alt text][image7]

A sliding window search is done to find lane pixels.

![alt text][image8]

Then I fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

For subsequent frames, lane identification was limited to the previously identified region.

(**Note**: Some attempts were made to do sanity check and use best fit when needed. However, it did't not work out well and that code is removed.)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #16 through #24 in my code in `lane.py`. It's essentially a refactored code from the lecture notes.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #147 through #183 in my code in `lane.py` in the function `draw_lane()`.  Here is an example of my result on test images:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_result.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I have mainly followed the lecture video and notes with some modification like threshold param etc. and works fine with `project_video.mp4`.

However, the algorithm is not generalized and does not work well with different lighting condition, blurred/absent lane marking, shadow and sharp curves.

The following measures can be taken to make it more robust:

- To combine other thresholds from different color space, use larger kernel size
- Store recent fits and use average fit when lane can't be identified confidently.