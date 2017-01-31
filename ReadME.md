# P4-AdvancedLaneDetection


## Synopsis
In `Project #4 - Advanced Lane Detection`, we used image processing techniques and color gradient manipulations to detect lanes on a road. The idea of this project was to be able to detect lanes on a road with some advanced techniques we learned about, including thresholding and perspective transform. With thresholding, we were able to cancel out a lot of the uneeded information in the image and were able to focus just on the lanes. With perspective transform, we were able to better calculate the lane lines by warping our image to a birds-eye-view. With this view, we could see the lanes parallel to each other, measure the curvature of the lanes, and identify the points on each lane more accurately see we were looking down on the lanes from above.

We began by calabrating our camera using chessboard images and were able to generate values we could use on our road images to improve image distortion. Next, we created a gradient threshold using sobel derivative gradients and color channel threshold. We then took an image of a road, undistorted it using our values from our camera calibration, applied our gradient to it, and performed an image transform to warp the image to a birds-eye-view. We then detected the left and right lanes of this image using a sliding window technique to detect points over each lane and drew a polynomial through these points. Finally, we warped the image back from brids-eye-view to the normal view, and were able to transform the lane and display it back on the original image. Once we had tested our pipeline, we were able to apply it to a video of a car driving on a highway and succesfully were able to detect the lane for each frame.


## Calibration and Distortion 
(Code for this section can be found in the function `get_points()` on line 77 of `P4.py`)

Our first step was to use images of chessboards to calibrate our camera. We looped throuhg images of chessboards provided in the `camera_cal` folder and detected the corners using openCV function `findChessBoardCorners()`. We provided this function with the image and its dimensions (9x6) and it returned the 2D points in the given plane. We stored these, along with the 3D points in real world space, for each of the images. You can see this process on line # of P4_Lanes.py. `objp` is the real world representation of the chess board in (x,y,z) coordinates. Since the chessboard is 2D, we initialized z=0. For each image that detects corners, we create a new `objp` and add it to my `objpoints` list, while adding the detected corners to the `imgpoints` list. 

With these values, we were able to compute the camera calibration and distortion coefficents with the help of two very useful openCV functions: `cv2.calibrateCamera()` and `cv2.undistort`.  `cv2.calibrateCamera()` allowed us to finds the intrinsic and extrinsic parameters of the camera from several views of a calibration pattern. We feed this function our `objpoints` and `imgpoints` and it returns to us our camera matrix and distortion coefficients. With these values, we are able to undistort our test image. We use `cv2.undistort()` to undistort our image with these values and get this result(saved in `output_images/chesss_undistorted.jpg`):

![alt tag](https://s24.postimg.org/ldo872b1x/Screen_Shot_2017_01_28_at_2_50_58_PM.jpg)

As you can see, `cv2.undistort()` has helped compensate for radial and tangential distortion of the chessboard image.

## Detecting Lanes in Images

The distortion and camera matricies used with the chess boards were stored in a pickle file and would later help us quickly calibrate our road images, since they were of the same dimension. We also used thresholding to develop an image in color channels that would help us discern lanes from other pixels, and warped our image into a birds-eye-view over the lane markers. Finally, we detected points on the lane, plotted a polynomial over the lane, and rewarped the lines back onto the original image to see if it was able to lay over the correct lanes. In the following sections, I describe how I applied each of these steps on one of our test images given to us for this project. You can find this image in `test_images/test1`, which going forward will be referred to as "the test image".

### Undistorting Image
(Code for this section can be found on line 423 of `P4.py`)

Below is an example of one of our test images after applying undistortion to it, using the distortion and camera matricies developed in the previous section(saved in `output_images/test1undist.png`):

![alt tag](https://s30.postimg.org/683q96m69/test1undist.png)

We again used `cv2.undistort()` on the test image, using the distortion and camera matricies previously obtained.

### Gradient and Color Thresholding
(Code for this section can be found in the function `get_thresh()` on line 113 of `P4.py`)

The next step involved generating a binary image to help us identify lanes. To get this image, we used gradient and color thresholds to 
isolate lane lines in the image. After experimenting with different colors, we found that the s threshold was the best option for color thresholding to isolate all of the lanes in our image. The important step here was to make sure we were able to detect both the white lanes as well as yellow lanes, while canceling out other colors to make sure that they did not show up as outlier pixels when we were ready to detect the lane pixels. The s threshold met all of these requirements. We also played around with thresholding of sobelx and when we combined both, we were able to get a binary image that was able to detect all lanes and cancel out a lot of the noise we were seeing before (saved in `output_images/test1binary.png`): 

![alt tag](https://s30.postimg.org/v7ab0j241/test1binary.png)


### Perspective Transform
(Code for this section can be found in the function `perspective_transform()` on line 216 of `P4.py`)

We performed a perspective transform on our binary image to be able to see our lanes from a birds-eye-view. The most important step in this process is to choose our source and destination points for the transform very carefully. A perspective transform uses these points to warp the image and effectively drag this image towards or away from the camera to change the perspective. Some of the benefits of looking at the road image from a bird's eye view are that we are able to observe if the lanes are parallel and the curvature of the lanes, which we will discuss later. The source points, defined as `src` are the four points we select that we want to 'grap' from the road, which should cover the lanes we care about. The destination points, defined as `dist`, are the four points where we want these src points to end up in our warped image. With some trial and error, we were able to calculate some points which we thought best captured the brids-eye-view of the lane with the lanes looking as parallel as possible to each other, and enough of the lane appearing so that we could identify points on each of the two lanes to draw a polygon over. We were able to produce a set of 4 points that translated over all of our test images:

```
src = np.float32(
        [[(img_size[0] * 3/16) - 20, img_size[1]],
         [(img_size[0] - 60), img_size[1]],
         [(img_size[0]/2 + 110), img_size[1] * 2/3],
         [(img_size[0]/3 + 125), img_size[1] * 2/3]])

 dst = np.float32(
        [[(img_size[0] * 3/16), img_size[1]],
         [(img_size[0] * 13/16), img_size[1]],
         [(img_size[0] * 13/16), img_size[1]/2 - 60],
         [(img_size[0] * 3/16), img_size[1]/2 - 60]])
 ```
 The points appeared as follows for my test images: 
 
![alt tag](https://s23.postimg.org/r4iggdvuz/Screen_Shot_2017_01_29_at_1_38_01_PM.jpg)
 
 I was then able to use `cv2.getPerspectiveTransform(src,dst)` to calculate the top down view. This function calculates the perspective transform of an image given 4 pairs of points, mapping each pair of source points (`src`), to its corresponding destination points (`dst`) (saved in `output_images/undistorted_bird_test1.png`):
 
 ![alt tag](https://s24.postimg.org/dn5n6jo39/undistorted_bird_test1.png)

### Finding Lane Pixels: The Sliding Door
(Code for this section can be found in the function `collect_points()` on line 278 of `P4.py`)

To find the pixels on the lanes, we utilized a sliding window method. We took the histogram of the bottom half of the image and essentially sliced the image into 9 slices. We then applied the sliding window method on each lane. We would begin by identifying the center of the lane by looking at the peak in the histogram. We would then make the center of our first window at the peak of the histogram. For each slice, we create a window for the bottom slice of our lane, making the center of the window this peak. The windows width would be 1/10th of the image's width, and its height would be 1/9th of the image's height. Thus, we would be using 9 windows for each lane, meaning 9 slices of each lane. With this window created for the bottom slice, we took the histogram of the pixels detected in the window and stored the (x,y) coordinates of the peak to our image points for the frame. We then moved one window up and made sure to use the peak of the previous window as the starting center for the new window, giving us a headstart on finding pixels in the following slice. If no points were detected in a window, we would skip this slice and not store any points for it. The center of the following window would be the "peak points" of the window prior to this failed window. Below, you can see an example of the windows and points generated for `test_images/test1.png` (saved in `output_images/test1_windows.png` and `output_images/test1_points.png`):

![alt tag](https://s23.postimg.org/vedmxg90r/test1_windows.png)

![alt tag](https://s23.postimg.org/v0caxuox7/test1points.png)


### Plotting Polynomial
(Code for this section can be found in the function `get_curvature()` on line 159 of `P4.py`)

Once this was done, we were able to take the detected points and plot a 2nd order polynomial over the lanes. This process can be seen in the function `get_curvature()` on line (saved in `output_images/test1poly.png`)

![alt tag](https://s23.postimg.org/xge4bp6zv/test1poly.png)

### Radius of Curvature and Position of Vehicle
(Code for this section can be found in the function `get_curvature()` on line 159 of `P4.py`)
We also were able to calculate the radius of curvature for each of the lane polynomials using the following equation:

![alt tag](https://s29.postimg.org/42cbl5vpj/Screen_Shot_2017_01_29_at_3_01_51_PM.jpg)

Where A and B are the first and second coefficients of the polynomial, respectively. Once we fit polynomials to each of the lanes, we were able to calculate the radius of curvature for both left and right lanes.

However, this value was actually in pixel space, so to transform it to real world space, we needed to redo our calculations, this time correcting for x and y scaling, using the conversion factors below:

```
ym_per_pix = 30/720
xm_per_pix = 3.7/700
```

We also calculated the offset of the car from the center of the lane. As we were told to assume that the camera was mounted at the center of the car, we calculated 720, which was half the width of the image, to be the position of the camera. We then calculated the mean of the centers of the two lane lines to be the center of the lane, and thus their difference was the offset of the car. To transform this from pixels to kilometers, we converted the value by multiplying it by our meters per pixel conversion coefficient in x dimension:

`offset = (camera_center - (left_peak+right_peak)/2 ) * xm_per_pix`

### Inverse Transform
(Code for this section can be found in the function `display_rewarp()` on line 363 of `P4.py`)

With our lanes identified and our polynomials drawn, we used an inverse perspective transform to warp our birds-eye-view image  back to the original view. We did this by reversing the the order of the source and destination points used when we first got our perspective transform:

`Minv = cv2.getPerspectiveTransform(dst, src)`

After recasting our left and right points into usable format for the `cv2.fillPoly()` function, we drew our lanes onto a warped blank image and used the previous `cv2.warpPerspective` to warp the drawn lines from birds-eye to the original view. Finally, we took this re-warped lane and combined it with the original image. Below, you can see an example of this, with the curvature for each lane displayed as well (saved in `output_images/test1_withtext.png`):

![alt tag](https://s30.postimg.org/4nsqvuhdd/withtext2.jpg)



## Testing Pipeline on Video
(Code for this section can be found on lines 434-450 of `P4.py`)

You can see a demo of my model [here] (https://youtu.be/G5fgP2rgBsE)

Once we tested on all of the test images provided, it was time to try out our pipeline on the challenge video. On first try, we noticed some scenarios that were distorting our lanes, such as shadows, changing of color of the road, and random outliers. To combat these changes, we created a `Line()` class that stored the information about lines that were detected in previous frames. This way, when we were about to draw a new lane, we were able to refer to data collected from previous lanes and decide whether or not the lane that we were detecting in this frame was valid or not. With this learning functionality in place, we were able to implement a few techniques that really helped filter out the bad lines that were being drawn in our frames


### (1) Storing Center of Lane
For each lane detected in a frame, we would go through our usual process of checking the histogram for the lane's, which would be the center of the lane. To chek if the lane in a frame was beginning its sliding window at an irregular point (i.e. had miscalculated the center of the lane), we would compare the current detected center with the center of the previous lane, which we would have saved in `Line.peak`. If the difference was > 100, we rejected this peak and instead, set the center of the lane at the previous peak. This greatly helped choppiness in the right lane, which would at times detect outliers in its bottom slice that would move the detected point too far to the left or right, making the drawn area spill into the adjacent lane.

### (2) Increasing Window Width
At each slice, we would take a histogram of the points detected in the window. Inititally, if we came a cross a window that had no pixels detected, we we would skip this slice or store the same point from the previous window. This became a problem when we were presented a curved lane where the window was not wide enough to detect the pixels at a critical slice where the lane began to curve. This would have a chain effect on the following windows, and we would end up storing points for blank spaces and the polynomial we would draw would dip out of the lane completely. We then implemented a new technique, where instead of move forward after we saw a window that did not detect any pixels, we would loop one more time on this slice with a window that covered half of the size of the image (0-720 for left, 720-1280 for right). This way, we guaranteed ourselves that we would be able to detect any pixels, because we increased our window width. Below is a raw image of a lane where the right lane had to increase the size of its window for the second to last slice, because the original sized window was unable to detect the pixels (saved in `output_images/test1_larger_window.png`):

![alt tag](https://s23.postimg.org/rijypl0i3/test1_larger_window.png)



## Discussion

The advantages of using gradient and color thresholding, plus correct image warping, were a big part of the success of this project. Playing around wiht the thresholding until I was finally able to get a decent image was tiresome, but in the end it made a really big difference in filtering out outlier pixels around the lane. Image warping, specifically choosing the correct src/dst points, was also a trial-and-error process but when we were able to get points that we were comfortable with, we were able to succesfully produce parallel lines that contained detectable lane pixels. 

One specific area we did not utilize in our algorithm was the curvature. Although this algorithm was able to produce a sufficient result for the video we were given, it may run into problems under different circumstances, in which case taking the curvature into account when deciding to accept or reject a lane may be needed. Also, the thresholding we applied for this project was effective in filtering out outliers on this freeway, but there may potentially be roads that are less forgiving in terms of objects or colors that would disrupt pixel detection.

Lastly, the sliding window method proved to be the true key to detecting the lanes. After never having heard of this approach before, we were able to develop an accurate model of a sliding window over each lane and when it came time to tune the lane detections, this was where most of the enhancments were done. Storing the pervious peak helped identify where the following lane should start, rejecting empty windows helped filter out outliers, and extending the window when nothing was detected helped detect a sharp curve in a slice that may not have been caught with the original window.

We are really proud of the project, and although it was tideous to combine all of these different methods and constantly tune thresholds and window detecting methods, these advanfced methods helped push this model to a functionally sound pipeline.
