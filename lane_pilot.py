# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import argparse

from zmcrobot import ZMCRobot
#from combined_thresh import combined_thresh
#from line_fit import line_fit, viz2, calc_curve, final_viz

import sys
from time import time


WINDOW_NAME = 'Line Pilot'

def transform_matrix_640():
	src = np.float32(
		[[11,479],
		[231,100],
		[407,100],
		[629, 479]])

	dst = np.float32(
		[[100, 479],
		[100, 0],
		[539, 0],
		[539, 479]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	return m, m_inv

def transform_matrix_320():
	src = np.float32(
		[[5,239],
		[110,50],
		[203,50],
		[315, 239]])

	dst = np.float32(
		[[50, 239],
		[50, 0],
		[269, 0],
		[269, 239]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	return m, m_inv


def perspect_transform_matrix():
	src = np.float32(
		[[11,479],
		[231,100],
		[407,100],
		[629, 479]])
        
	dst = np.float32(
		[[100, 479],
		[100, 0],
		[539, 0],
		[539, 479]])

	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	return m, m_inv
    
        
def image_left( image ):
	height = image.shape[0]
	width = image.shape[1]
	mask = np.zeros_like( image )
	cv2.rectangle(mask, (0, 0), (int(width/2), height), 255, -1 )
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def image_right(image ):
	height = image.shape[0]
	width = image.shape[1]
	mask = np.zeros_like( image )
	cv2.rectangle(mask, (int(width/2), 0), (width, height), 255, -1 )
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

#get the line start x0 according to the dot histogrtam
def line_start_x0( image ):
	height = image.shape[0]
	width = image.shape[1]
	histogram = np.sum(image[height - 30:,:], axis=0)
	x0 = 0
	cnt = 0
	for idx in range(0, width ):
		if histogram[idx] != 0 :
			x0 = x0 + idx
			cnt = cnt + 1
	if cnt != 0 :
		x0 = x0 / cnt
	return x0


def make_coordinates(height, line_parameters ):
    slope, intercept = line_parameters
    y1 = height
    y2 = 0 # int(y1 *(3/5))
    x1 = int ((y1 - intercept)/slope)
    x2 = int ((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slop_intercept(height, lines ):
	line_fit = []
	for line  in lines:
		x1,y1,x2,y2  = line.reshape(4)
		parameters = np.polyfit((x1,x2), (y1,y2), 1 )
		slope = parameters[0]
		intercept = parameters[1]
		line_fit.append((slope, intercept))
		line_fit_average = np.average( line_fit, axis = 0)
	line = make_coordinates(height, line_fit_average)
	return line
 

def average_line(height, start_x,  lines ):
	theta =0.0
	cnt = 0
	x0 = 0.0

	if lines is  None:
		return None, 0, 0

#	print('count avg line: ')
	for line in lines:
		x1,y1,x2,y2 = line.reshape(4)
		atheta = math.atan2(y2-y1, x2-x1)
		if( atheta < 0 ):
			atheta = math.atan2(y1-y2, x1-x2)
#		if abs( theta ) < ( 5.0 * np.pi / 180) :  #ignore the 
#			continue
		if  atheta  >  0.18 and atheta < 2.9  :
			xc = cross_x(x1, y1, x2, y2, height-1  )
			if( xc >=start_x - 25 and xc <= start_x + 25  ):
				theta =theta + atheta
				cnt = cnt + 1
				x0 = x0 + xc
		#	print( 'xc: %0.2f theta: %0.3f' %( xc, atheta ) )
	#	else :
	#		print( 'x1: %0.2f theta: %0.3f' %( x1, atheta ) )
	if cnt == 0 :
	#	print( 'no line found')
		return None, 0, 0

	avg_x  = x0 / cnt
	avg_theta = theta / cnt

	avg_theta = math.atan2(math.sin( avg_theta), math.cos( avg_theta ) )
#	print('x0:%0.2f theta: %.3f' % (avg_x, avg_theta ) )
	x1 = int( avg_x )
	y1 = height - 1
	x2 = int(  x1 - 500.0 * math.cos( avg_theta ))
	y2 = int( y1 - 500.0 * math.sin( avg_theta ))
	aline = np.array([x1,y1,x2,y2])
#	print( aline )
	return aline, avg_theta, x1



def cross_x(x1, y1, x2, y2, h ):
	if x1 - x2 ==  0 :
		return x1
	a = (y1-y2)/(x1 -x2)
	b = y1 - a * x1
	x = (h  - b )/a
	return x

def canny( image ):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 25, 150)
	return canny

def auto_canny(image, sigma=0.33):
	v = np.median( image )
	lower = int(max(0, (1.0-sigma)*v))
	upper = int(min(255, (1.0 + sigma)*v))
	edged = cv2.Canny( image, lower, upper )
	return edged

def display_lines(image, lines, line_color, line_width = 1, line_image = None  ):
	if line_image is None:
		line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1,y1,x2,y2 =  line.reshape(4)
			cv2.line( line_image, (x1,y1), (x2, y2), line_color, line_width )
	return line_image


def hough_lines( image ):
	width = image.shape[1]
	height = image.shape[0]
	left_image = image_left( image  )
	right_image = image_right(image )
	left_lines = cv2.HoughLinesP(left_image, 2, np.pi/180, 100, np.array([]), minLineLength=50, maxLineGap=5)
	right_lines = cv2.HoughLinesP(right_image, 2, np.pi/180, 100, np.array([]), minLineLength=50, maxLineGap=5)

	line_start_left_x = line_start_x0( left_image )
	line_strart_right_x = line_start_x0( right_image )

	#	print('x0-l: %d x0-r: %d ' % (line_start_left_x, line_strart_right_x))
	avg_left_line, avg_theta_left, x0_left = average_line(height,  line_start_left_x,	left_lines )
	avg_right_line, avg_theta_right,  x0_right = average_line(height, line_strart_right_x,	right_lines )

	lines = []
	if left_lines is not None:
		for line in left_lines :
			lines.append( line )
	if right_lines is not None:
		for line in right_lines :
			lines.append( line )

	not_rec = False
	if avg_left_line is  None :
		not_rec = True
	else:
		lines.append( avg_left_line)

	if avg_right_line is None:
		not_rec = True
	else:
		lines.append( avg_right_line )

	target_line = None
	if not_rec == False:
		x1 = width/2
		y1 = height - 1
		avg_theta = (avg_theta_left + avg_theta_right)/2
		x2 = int(x1 - 300.0 * math.cos(avg_theta ))
		y2 =  int(x1 - 300.0 * math.sin(avg_theta))
		x1 = int(x1)
		target_line = np.array([x1,y1,x2,y2])
		lines.append(target_line  )
	d_center = line_start_left_x - width + line_strart_right_x
	return lines, target_line, avg_theta, d_center 



def line_fit( image ):
	"""
	Find and fit lane lines
	"""
	width = image.shape[1]
	height = image.shape[0]
	# Take a histogram of the bottom half of the image
	histogram = np.sum(image[height//2:,:], axis=0)
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[0:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint
	# Choose the number of sliding windows
	nwindows = 8
	# Set height of windows
	window_height = np.int(height/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 30
	# Set minimum number of pixels found to recenter window
	minpix = 30
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = height - (window+1)*window_height
		win_y_high = height - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	lines = []
	
	# x0 = leftx[0]
	# y0 = lefty[0]
	# for i in range( len(leftx) - 1 ):
	# 	x1 = leftx[i+1]
	# 	y1 = lefty[i+1]
	# 	lines.append([x0,y0,x1,y1])
	# 	x0 = x1
	# 	y0 = y1

	# x0 = rightx[0]
	# y0 = righty[0]
	# for i in range( len(rightx) - 1 ):
	# 	x1 = rightx[i+1]
	# 	y1 = righty[i+1]
	# 	lines.append([x0,y0,x1,y1])
	# 	x0 = x1
	# 	y0 = y1

	if lefty is None or righty is None:
		return None, None, 0, 0

	left_fit = np.polyfit(lefty, leftx, 1)
	right_fit = np.polyfit(righty, rightx, 1)

	y0 = height -1
	x0 = int(left_fit[0] * y0 + left_fit[1])
	y1 = 50
	x1 = int(left_fit[0] * y1 + left_fit[1])
	aline = np.array([x0,y0,x1,y1])
	lines.append(aline)

	theta_l = math.atan2(y1 - y0, x1 - x0 )

	y0 = height -1
	x0 = int(right_fit[0] * y0 + right_fit[1])
	y1 = 50
	x1 = int(right_fit[0] * y1 + right_fit[1])

	aline = np.array([x0,y0,x1,y1])
	lines.append(aline)

	theta_r = math.atan2(y1 - y0, x1 - x0 )
	avg_theta = ( theta_l + theta_r )/2

	x0 = int(( leftx_base + rightx_base ) /2)
	y0 = height - 1
	x1 = int( x0 + 200 * math.cos( avg_theta ))
	y1 = int(y0 + 200 * math.sin( avg_theta ))

	# x1 = int(200 * math.cos( avg_theta ))
	# y1 = int(200 * math.sin( avg_theta ))

	aline = np.array([x0,y0,x1,y1])
	lines.append(aline)


	# Fit a second order polynomial to each
	#left_fit = np.polyfit(lefty, leftx, 2)
	#right_fit = np.polyfit(righty, rightx, 2)
	
	return lines, aline, avg_theta,  int(x0 - width/2)




def line_pilot( cap, width, height ):
	with open('camera_cal_640_480.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']
	zmcRobot = ZMCRobot()

	m, m_inv =transform_matrix_640()
	if width == 320:
		m, m_inv = transform_matrix_320()
	# height = 480
	# width = 640
	while True:
		if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
			break #check to see if the user has closed the window

		start = time()
		_, image = cap.read() #grap the next image frame
		undis_image = cv2.undistort(image, mtx, dist, None, mtx)
		canny_image = canny( undis_image )
		wraped_image = cv2.warpPerspective(canny_image, m, (width, height))

	#	lines,  target_line, line_theta,  d_center   = hough_lines( wraped_image )
		lines,  target_line, line_theta,  d_center   = 	line_fit(wraped_image)
		line_image = display_lines( image, lines, (0,0,255), 1, None )
		
		ctrl_theta = -np.pi/2 - line_theta
		if target_line is not None:
			zmcRobot.drive_car(0.09, ctrl_theta )

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(line_image, m_inv, (width, height))
	# Combine the result with the original image
		result_image = cv2.addWeighted(undis_image, 1, newwarp, 0.3, 0)
		elapsed = time() - start
		label = 'w: %.3f dc: %d;t:%.2f' % (ctrl_theta, d_center, elapsed*1000)
		result_image = cv2.putText(result_image, label, (30,40), 0, 0.7, (128,0,255), 2, cv2.LINE_AA)

		cv2.imshow(WINDOW_NAME, result_image )
		key = cv2.waitKey(10)
		if key == 27: # ESC key: quit program
			break
	zmcRobot.shutdown()

def open_cam_onboard(width, height, sensor_id):
	gst_str = ('nvarguscamerasrc '
                   'sensor-id={} ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)3264, height=(int)1848, '
                   'format=(string)NV12, framerate=(fraction)20/1 ! '
                   'nvvidconv flip-method=0 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(sensor_id, width, height)
	print( gst_str )
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)



def open_window(width, height):
	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(WINDOW_NAME, width, height)
	cv2.moveWindow(WINDOW_NAME, 0, 0)
	cv2.setWindowTitle(WINDOW_NAME, 'Line Pilot')


def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--sensor_id', dest='sensor_id', help='sensor id for csi camera 0/1', default=0)

    parser.add_argument('--width', dest='image_width',
                        help='image width [1920]',
                        default=1280, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=720, type=int)
    parser.add_argument('--file', dest='file_name',
                        help='save file name [capfile]',
                        default='capfile' )
    args = parser.parse_args()
    return args



def main():
	args = parse_args()
	print('Called with args:')
	print(args)
	print('OpenCV version: {}'.format(cv2.__version__))
	cap = open_cam_onboard(args.image_width,
                               args.image_height, args.sensor_id )

	if not cap.isOpened():
		sys.exit('Failed to open camera!')
	open_window(args.image_width, args.image_height)
	line_pilot( cap,  args.image_width, args.image_height )
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

