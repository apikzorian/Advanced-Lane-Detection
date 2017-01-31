import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
from matplotlib.lines import Line2D
from moviepy.video.io.VideoFileClip import VideoFileClip

count = 0


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.count_x = 0

        self.peak = None

    def add_x(self, x):
        self.recent_xfitted.append(x)
        self.count_x += 1
        length = len(self.bestx)
        if length == 0:
            self.bestx = x
        else:
            new_best = []
            for i in range(0, len(x)):
                new_best.append(((self.bestx[i] + x[i])/self.count_x))
            self.bestx = new_best

    def wipe_line(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None



def get_points():
    '''
    Calibrates camera using chessboard image files in camera_cal folder

    :return:
    objpoints - Collection of 3d points in real world space
    imgpoints - Collection of corners detected in chessboard images
    '''
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            write_name = 'corners found for ' + fname
        else:
            write_name = 'corners NOT found for ' + fname
    return objpoints, imgpoints


def get_thresh(img):
    '''
      Uses combination of Sobel gradient and color thresholding to create binary version of input image

      :param img: Image frame of lane to be thresholded
      :return: Binary image after color and gradient thresholding applied
      '''

    # Acquires s_channel of image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 90
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1


    return combined_binary


def get_curvature(img, yvals, leftx, rightx):
    '''
    Calculate the curvature of each lane and the car's deviation from the center of the lane
    :param img: Bird's eye view of image
    :param yvals: y values of lanes
    :param leftx: x values of left lane
    :param rightx: x values of right lane
    :return:
    '''


    left_fit = np.polyfit(yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals ** 2 + left_fit[1] * yvals + left_fit[2]
    right_fit = np.polyfit(yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals ** 2 + right_fit[1] * yvals + right_fit[2]

    y_eval = np.max(yvals)
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    left_fit_cr = np.polyfit(yvals * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(yvals * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])

    # Calculate center of lane
    center = (left_curr.peak + right_curr.peak) / 2
    offset = (center - 720) * (3.7 / 700)
    left_curr.line_base_pos = right_curr.line_base_pos = round(offset,3)

    if not left_line.detected:
        left_line.radius_of_curvature = left_curr.radius_of_curvature = round(left_curverad,2)
        left_line.current_fit = left_fit
        left_line.line_base_pos = left_curr.line_base_pos
        left_line.detected = True
    else:
        left_curr.radius_of_curvature = left_curverad

    if not right_line.detected:
        right_line.radius_of_curvature = right_curr.radius_of_curvature = round(right_curverad,2)
        right_line.current_fit = right_fit
        right_line.line_base_pos = right_curr.line_base_pos
        right_line.detected = True
    else:
        right_curr.radius_of_curvature = right_curverad


    # Calculate center of lane
    center = (left_curr.peak + right_curr.peak) / 2
    offset = (center - 720) * (3.7 / 700)
    left_curr.line_base_pos = right_curr.line_base_pos = round(offset,3)

    return yvals, left_fitx, right_fitx


def perspective_transform(img):
    '''
    Warp image to Bird's-eye view using cv2.getPerspectiveTransform. Establish source and destination points
    for perspective transform
    :param img: Original image
    :return: Warped image and inverse transform
    '''

    img_size = (img.shape[1], img.shape[0])

    # Access undistorted image
    with open("camera_cal/camera_dist_pickle.p", mode='rb') as f:
        pfile = pickle.load(f)
    mtx = pfile["mtx"]
    dist = pfile["dist"]
    undst = cv2.undistort(img, mtx, dist, None, mtx)

    # Compute source and destination points to warp to
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

    # Perform perspective transform, warp image, and perform inverse transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, Minv

def get_hist(img, y_bins, peak, i, window_width):
    '''
    Get histogram of given slice using provided parameters
    :param img: Birds-Eye View image
    :param y_bins: # of windows
    :param peak: Peak of current lane
    :param i: Slice #
    :param window_width: Width of detection window
    :return: Histogram generated by parameters
    '''

    yMax = y_bins[i + 1]
    yMin = y_bins[i]
    xMin = peak - window_width / 2

    xMax = xMin + window_width
    if xMax > 1200:
        xMax = 1200
    if xMin < 150:
        xMin = 150
    right_img = img[yMax:yMin, xMin:xMax]
    hist = np.sum(right_img, axis=0)

    return hist, xMin, xMax, yMin, yMax

def collect_points(img):
    '''
    Collect pixels detected in birds-eye-view image using sliding window method
    :param img: Birds-eye-view image
    '''
    # Number of slices
    num_y_bins = 9

    # Width of each window of each slice
    window_width = img.shape[1] / 10

    # Histogram over bottom half of image
    histogram = np.sum(img[img.shape[0] / 2:, :], axis=0)
    x_split = len(histogram) / 2

    # Two peaks of each lane, left and right. np.argmax() returns index of highest point in histogram
    left_peak = np.argmax(histogram[:x_split])
    right_peak = np.argmax(histogram[x_split:]) + x_split

    # If difference between current peak and previous peak is > 100, we disregard current peak and start at the
    # peak from the previous window
    if (right_line.peak != None):
        right_peak_diff = right_line.peak - right_peak
        if(right_peak_diff > 100):
            right_peak = right_curr.peak

    left_curr.peak = left_peak
    right_curr.peak = right_peak
    y_bins = np.linspace(img.shape[0], 0, num_y_bins)

    left_lane = []
    right_lane = []
    ally = []

    '''
    For each slice, we take the histogram of the window and store peak x and corresponding y values
    '''

    for i in range(num_y_bins - 1):

        hist_right, xMin, xMax, yMin, yMax = get_hist(img, y_bins, right_peak, i, window_width)
        first_hist_right = hist_right
        if hist_right.size != 0 and max(hist_right) != 0:
            right_peak = np.argmax(hist_right) + xMin
        else:
            new_width = img.shape[1]/2
            hist_right, xMin, xMax, yMin, yMax = get_hist(img, y_bins, right_peak, i, new_width)
            if hist_right.size == 0 or max(hist_right) == 0:
                hist_right = first_hist_right
            else:
                right_peak = np.argmax(hist_right) + xMin

        right_lane.append(right_peak)
        ally.append(((yMax + yMin) / 2))


        hist_left, xMin, xMax, yMin, yMax = get_hist(img, y_bins, left_peak, i, window_width)
        first_hist_left = hist_left
        if hist_left.size != 0 and max(hist_left) != 0:
            left_peak = np.argmax(hist_left) + xMin
        else:
            new_width = img.shape[1]/2
            hist_left, xMin, xMax, yMin, yMax = get_hist(img, y_bins, left_peak, i, new_width)
            if hist_left.size == 0 or max(hist_left) == 0:
                hist_left = first_hist_left
            else:
                left_peak = np.argmax(hist_left) + xMin

        left_lane.append(left_peak)

    left_curr.allx = left_lane
    right_curr.allx = right_lane

    # Use current left and right lane values to obtain curvature and fitted x and y values for lanes

    yvals, left_fitx, right_fitx = get_curvature(img, np.asarray(ally), np.asarray(left_lane), np.asarray(right_lane))

    left_line.add_x(left_lane)
    left_line.peak = left_curr.peak

    right_line.add_x(right_lane)
    right_line.peak = right_curr.peak

    return yvals, left_fitx, right_fitx, left_curr.radius_of_curvature, right_curr.radius_of_curvature

def display_rewarp(yvals, left_fitx, right_fitx, image, warped, Minv, undist, left_curve, right_curve):

    '''
    Warp image back to original view, display lane, and print curvature and position error values. Return image
    with all of these factors projected over lane

    '''

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    left_text = "Left curvature " + str(left_curve)
    right_text = "Right curvature " + str(right_curve)
    offset_text = "Pos Error =" + str(left_curr.line_base_pos)
    font = cv2.FONT_HERSHEY_COMPLEX

    cv2.putText(result, left_text, (30, 45), font, 1, (255, 0, 0), 2)
    cv2.putText(result, right_text, (30, 90), font, 1, (255, 0, 0), 2)
    cv2.putText(result, offset_text, (30, 135), font, 1, (0, 255, 0), 2)

    global count

    # Save frames for debugging purposes
    cv2.imwrite("out_frames_curve/frame%d.jpg" % count, result)  # save frame as JPEG file
    count += 1

    if count == 225:
        print("esh")
    return result


def process_frame(img):
    '''
    Take frame from video, apply undistortion, thresholding, and perform perspective transform. After that, collect
    lane pixels from birds-eye-view image and draw polynomial. Finally, perform inverse transform on birds-eye-view
     image and lay polynomial over original image
    :param img: Frame from video
    :return: Same frame with lane drawn on it
    '''


    # Erase previous left_curr and right_curr
    left_curr.wipe_line()
    right_curr.wipe_line()
    # Undistort image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Get color and
    thresh_img = get_thresh(img)
    top_down, Minv = perspective_transform(thresh_img)
    yvals, left_fitx, right_fitx, left_curve, right_curve = collect_points(top_down)
    result = display_rewarp(yvals, left_fitx, right_fitx, img, top_down, Minv, undist, left_curve, right_curve)
    return result


# 4 instances of Line class, two for current lanes (left and right), and two for storing information about all
#  detected lines
left_line = Line()
right_line = Line()
left_curr = Line()
right_curr = Line()

video_output = 'video_out_0129_2.mp4'
clip1 = VideoFileClip("project_video.mp4")

# Load calibration matricies
with open("camera_cal/camera_dist_pickle.p", mode='rb') as f:
    pfile = pickle.load(f)
mtx = pfile["mtx"]
dist = pfile["dist"]

# Perform image processing on each frame and save new video
video_clip = clip1.fl_image(process_frame)
video_clip.write_videofile(video_output, audio=False)
