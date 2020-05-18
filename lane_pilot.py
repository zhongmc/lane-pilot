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
from web_camera import WebController
#from combined_thresh import combined_thresh
#from line_fit import line_fit, viz2, calc_curve, final_viz

import sys
from time import time
from threading import Thread

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
	margin = 60
	# Set minimum number of pixels found to recenter window
	minpix = 15
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low =height - (window+1)*window_height
		win_y_high =height - window*window_height
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


	if lefty is None or righty is None:
		return None, None, None, 0, 0
	if len(lefty ) == 0 or len(righty ) == 0 :
		return None, None, None, 0, 0

	lines = []

#二阶拟合，求切线
	left_line, left_fit, left_x0, kl  = line_of_poly( leftx, lefty,   height-1, 80  )  # x = ay^2 + by + c
	lines.append( left_line )

	right_line, right_fit, right_x0, kr  = line_of_poly( rightx, righty,  height-1, 80  )
	lines.append( right_line )

#中间目标线
	k0 = (kl + kr)/2
	x0 = (left_x0 + right_x0)/2
	y0 = height - 1
	y1 = 80
	x1 = k0 * (y1 - y0) + x0
	avg_theta = math.atan2( y1 - y0, x1-x0 )
	x0 = int(x0)
	x1 = int( x1 )
	goal_line = np.array([x0, y0, x1, y1 ])
	lines.append( goal_line )
	return lines,  left_fit, right_fit, avg_theta, int(x0 - width/2)

#二阶拟合，求切线
def line_of_poly( xp, yp,  y0, y1 ):
	poly_fit = np.polyfit(yp, xp, 2)  #x = ay^2 + by + c
	x0 = poly_fit[0] *y0**2 + poly_fit[1] * y0 + poly_fit[2]
	k = 2* poly_fit[0] *y0 + poly_fit[1]
	x1 = int( k*(y1 - y0)  + x0 )
	x0 = int( x0 )
	return np.array([x0, y0, x1, y1] ), poly_fit, x0, k

zmcRobot = ZMCRobot()

def doDrive(angle, throttle ):
	zmcRobot.drive_car(throttle, angle  )
	pass

def line_pilot( cap, width, height ):

	cal_file = 'camera_cal' + str(width) + '-' + str(height) + '.p'
	with open(cal_file, 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']
	web = WebController()
	web.call_back = doDrive
	t = Thread(target=web.start, args=())
	t.daemon = True
	t.start()
  
	m, m_inv =transform_matrix_640()
	if width == 320:
		m, m_inv = transform_matrix_320()
	# height = 480
	# width = 640
	elapsed = 0
	ctrl_theta = 0
	d_center = 0

	while True:
		if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
			break #check to see if the user has closed the window

		start = time()
		ret, image = cap.read() #grap the next image frame
		if not ret:
			key = cv2.waitKey(20)
			if key == 27: # ESC key: quit program
				break
			continue

		undis_image = cv2.undistort(image, mtx, dist, None, mtx)
		canny_image = canny( undis_image )
		wraped_image = cv2.warpPerspective(canny_image, m, (width, height))

		lines,  left_fit, right_fit, line_theta,  d_center   = line_fit( wraped_image )
		line_image = np.zeros_like(image)
		if lines is not None:
			cnt = int(height / 10)
			ploty = np.linspace(0,  height -1, cnt  )
			left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
			right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
			for i in range(cnt - 1) :
				xl0 = int( left_fitx[i])
				xr0 = int(right_fitx[i])
				y0 = int(ploty[i])
				xl1 = int(left_fitx[i + 1])
				xr1 = int(right_fitx[i+1])
				y1 = int(ploty[i+1])
				cv2.line( line_image, (xl0,y0), (xl1, y1), (255, 255, 0), 10 )
				cv2.line( line_image, (xr0,y0), (xr1, y1), (255, 255, 0),  10  )

			x1,y1,x2,y2 =  lines[0].reshape(4)
			cv2.line( line_image, (x1,y1), (x2, y2), (0, 0, 255), 2 )
			x1,y1,x2,y2 =  lines[1].reshape(4)
			cv2.line( line_image, (x1,y1), (x2, y2), (0, 0, 255), 2 )
			x1,y1,x2,y2 =  lines[2].reshape(4)
			cv2.line( line_image, (x1,y1), (x2, y2), (255, 0, 0), 5 )
			ctrl_theta = np.pi/2 + line_theta
			zmcRobot.drive_car(0.09, ctrl_theta )

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(line_image, m_inv, (width, height))
	# Combine the result with the original image
		result_image = cv2.addWeighted(undis_image, 1, newwarp, 0.5, 0)
		label = 'w: %.3f dc: %d;t:%.2f' % (ctrl_theta, d_center, elapsed*1000)
		result_image = cv2.putText(result_image, label, (30,40), 0, 0.7, (128,0,128), 2, cv2.LINE_AA)
		web.update_image( result_image )

		elapsed = time() - start

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
		print( 'failed to open camera!')
		#sys.exit('Failed to open camera!')
	open_window(args.image_width, args.image_height)
	line_pilot( cap,  args.image_width, args.image_height )
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()

