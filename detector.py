import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import argparse
from helper_functions import *
my_parser = argparse.ArgumentParser()

my_parser.add_argument('path', type=str)
my_parser.add_argument('mode', type=str)

#Defining image processing method
global used_warped
global used_ret

def process_image(image):
    global used_warped
    global used_ret
    imgpoints, objpoints = collect_callibration_points()
    #Undistort image
    img = image.copy()
    image, mtx, dist_coefficients = cal_undistort(image, objpoints, imgpoints)
    if mode == 'debug':
        compare_images(img, image,image2_exp="camera callibration")
    # Gradient thresholding
    gradient_combined = apply_thresholds(image)
    if mode == 'debug':
        compare_images(img, gradient_combined,image2_exp="Gradient thresholding")
    # Color thresholding
    s_binary = apply_color_threshold(image)
    if mode == 'debug':
        compare_images(img, s_binary,image2_exp="Color thresholding")
    # Combine Gradient and Color thresholding
    combined_binary = combine_threshold(s_binary, gradient_combined)
    if mode == 'debug':
        compare_images(img, combined_binary,image2_exp="Gradient + Color thresholding")

    # Transforming Perspective
    binary_warped, Minv = warp(combined_binary)
    if mode == 'debug':
        compare_images(img, binary_warped,image2_exp="Transform Prespective")

    # Getting Histogram
    histogram = get_histogram(binary_warped)

    # Sliding Window to detect lane lines
    ploty, left_fit, right_fit = slide_window(binary_warped, histogram)

    # Skipping Sliding Window
    ret = skip_sliding_window(binary_warped, left_fit, right_fit)

    # Measuring Curvature
    left_curverad, right_curverad = measure_curvature(ploty, ret)

     # Sanity check: whether the lines are roughly parallel and have similar curvature
    slope_left = ret['left_fitx'][0] - ret['left_fitx'][-1]
    slope_right = ret['right_fitx'][0] - ret['right_fitx'][-1]
    slope_diff = abs(slope_left - slope_right)
    slope_threshold = 150
    curve_diff = abs(left_curverad - right_curverad)
    curve_threshold = 10000
    if "mp4" in path:
        if (slope_diff > slope_threshold or curve_diff > curve_threshold):
            binary_warped = used_warped
            ret = used_ret

    # Visualizing Lane Lines Info
    result = draw_lane_lines(image, binary_warped, Minv, ret)

    # Annotating curvature
    fontType = cv2.FONT_HERSHEY_SIMPLEX
    curvature_text = 'The radius of curvature = ' + str(round(left_curverad, 3)) + 'm'
    cv2.putText(result, curvature_text, (30, 60), fontType, 1.5, (255, 255, 255), 3)

    # Annotating deviation2
    deviation_pixels = image.shape[1]/2 - abs(ret['right_fitx'][-1] - ret['left_fitx'][-1])
    xm_per_pix = 3.7/700
    deviation = deviation_pixels * xm_per_pix
    direction = "left" if deviation < 0 else "right"
    deviation_text = 'Vehicle is ' + str(round(abs(deviation), 3)) + 'm ' + direction + ' of center'
    cv2.putText(result, deviation_text, (30, 110), fontType, 1.5, (255, 255, 255), 3)

    used_warped = binary_warped
    used_ret = ret

    return result

#Applying to Video
from moviepy.editor import VideoFileClip
from IPython.display import HTML

args = my_parser.parse_args()
global mode 
global path
mode = args.mode
path = args.path
if args.mode == 'debug':
    if "mp4" in path:
        output = 'result.mp4'
        clip = VideoFileClip(path)
        video_clip = clip.fl_image(process_image)
        video_clip.write_videofile(output, audio=False)
    else:
        output = "result.jpg"
        image = mpimg.imread(path)
        output_img = process_image(image)
        cv2.imwrite(output, output_img)
else:
    if "mp4" in path:
        output = 'result.mp4'
        clip = VideoFileClip(path)
        video_clip = clip.fl_image(process_image)
        video_clip.write_videofile(output, audio=False)
    else:
        output = "result.jpg"
        image = mpimg.imread(path)
        output_img = process_image(image)
        cv2.imwrite(output, output_img)
