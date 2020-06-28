# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import argparse

# from zmcrobot import ZMCRobot
# from web_camera import WebController
#from combined_thresh import combined_thresh
#from line_fit import line_fit, viz2, calc_curve, final_viz

import sys
from time import time
from threading import Thread

WINDOW_NAME = 'Line Pilot'


def transform_matrix_640():
	# src = np.float32(
	# 	[[0,479],
	# 	[221,160],
	# 	[419,160],
	# 	[639, 479]])
	src = np.float32(
		[[0,479],
		[165,240],
		[474,240],
		[639, 479]])

	dst = np.float32(
		[[100, 479],
		[100, 0],
		[539, 0],
		[539, 479]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	lane_width = 230
	return m, m_inv, src, lane_width

def transform_matrix160_320():
	src = np.float32(
		[[0, 239],
		[107,110],  
		[212,110], 
		[319, 239]])

	dst = np.float32(
		[[80, 239],
		[80, 0],
		[239, 0],
		[239, 239]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	lane_width = 125
	return m, m_inv, src, lane_width


def transform_matrix160_244():
	src = np.float32(
		[[0, 243],
		[78,100],  
		[168,100],
		[243, 243]])
	dst = np.float32(
		[[80, 243],
		[80, 0],
		[163, 0],
		[163, 243]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	lane_width = 50
	return m, m_inv, src, lane_width

###############################
def transform_matrix120_320():
	src = np.float32(
		[[0, 239],
		[102,100],  
		[225,100], 
		[319, 239]])
	dst = np.float32(
		[[80, 239],
		[80, 0],
		[239, 0],
		[239, 239]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	lane_width = 120
	return m, m_inv, src, lane_width


def transform_matrix120_244():
	src = np.float32(
		[[0, 243],
		[78,100],  
		[168,100],
		[243, 243]])
	dst = np.float32(
		[[80, 243],
		[80, 0],
		[163, 0],
		[163, 243]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	lane_width = 50
	return m, m_inv, src, lane_width



def transform_matrix( width, height, lens ):
	if lens == 120:
		if width == 320:
			m, m_inv, src, lane_width = transform_matrix120_320()
		elif width == 244:
			m, m_inv, src, lane_width = transform_matrix120_244()
	elif lens == 160:
		if width == 320:
			m, m_inv, src, lane_width = transform_matrix160_320()
		elif width == 244:
			m, m_inv, src, lane_width = transform_matrix160_244()
	else:
		m, m_inv, src, lane_width = transform_matrix120_320()

	return m, m_inv, src, lane_width

# functions for hough lines	   

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


def cross_x(x1, y1, x2, y2, h ):
	if x1 - x2 ==  0 :
		return x1
	a = (y1-y2)/(x1 -x2)
	b = y1 - a * x1
	x = (h  - b )/a
	return x

def average_line(height, start_x,  lines ):
	theta =0.0
	cnt = 0
	x0 = 0.0
	print('count avg line: ')
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
			print( 'xc: %0.2f theta: %0.3f' %( xc, atheta ) )
		else :
			print( 'x1: %0.2f theta: %0.3f' %( x1, atheta ) )
	if cnt == 0 :
		print( 'no line found')
		return None, 0, 0

	avg_x  = x0 / cnt
	avg_theta = theta / cnt

	avg_theta = math.atan2(math.sin( avg_theta), math.cos( avg_theta ) )
	print('x0:%0.2f theta: %.3f' % (avg_x, avg_theta ) )
	x1 = int( avg_x )
	y1 = height - 1
	x2 = int(  x1 - 500.0 * math.cos( avg_theta ))
	y2 = int( y1 - 500.0 * math.sin( avg_theta ))
	aline = np.array([x1,y1,x2,y2])
	print( aline )
	return aline, avg_theta, x1


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


def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
	"""
	Return the magnitude of the gradient
	for a given sobel kernel size and threshold values
	"""
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255
	gradmag = (gradmag/scale_factor).astype(np.uint8)
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 255

	# Return the binary image
	return binary_output


def sobel( image ):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	mag_bin = mag_thresh(blur, sobel_kernel=3, mag_thresh=(50, 255))
	return mag_bin


def canny( image ):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	#双边滤波，平滑去噪的同时很好得保存边沿
	# blur = cv2.bilateralFilter(gray, 11, 17,17)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 25, 150)
	return canny

def auto_canny(image, sigma=0.33):
	v = np.median( image )
	lower = int(max(0, (1.0-sigma)*v))
	upper = int(min(255, (1.0 + sigma)*v))
	edged = cv2.Canny( image, lower, upper )
	return edged

#horizen histogram top 4 peak position; used to get the horizen lines 
def horizen_peaks(image, peak_level ):
	histogram = np.sum(image, axis=1)
	avg = np.mean( histogram )
	travg = int( peak_level * avg)
	peaks = []
	for i in range(1, len(histogram)- 2 ):
		if histogram[i-1] < histogram[i] >histogram[i+1]  and histogram[i] > travg:
			peaks.append((i, histogram[i]))

	# print( peaks )

	if len(peaks) < 4:
		return None

	ps = np.array( peaks, dtype= np.int32)
	sid =  np.lexsort(-ps.T)
	ps = ps[ sid ]
	# print( peaks )
	# print( ps )

	idxs = [0, 0, 0, 0]
	idxs[0] = ps[0][0]
	idxs[1] = ps[1][0]
	idxs[2] = ps[2][0]
	idxs[3] = ps[3][0]
	# print( idxs )
	idxs.sort()
	return idxs

#二阶拟合，求切线
def line_of_poly( xp, yp, y,  y0, y1 ):
	poly_fit = np.polyfit(yp, xp, 2)  #x = ay^2 + by + c
	k = 2* poly_fit[0] *y + poly_fit[1]

	x0 = poly_fit[0] *y0**2 + poly_fit[1] * y0 + poly_fit[2]
	x1 = int( k*(y1 - y0)  + x0 )
	x0 = int( x0 )
	return np.array([x0, y0, x1, y1] ), poly_fit, x0, k


def line_fit_with_image( image ):
	width = image.shape[1]
	height = image.shape[0]
	# Take a histogram of the bottom half of the image
	histogram = np.sum(image[200:,:], axis=0)  #height//2
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
	margin = int(width/16)
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
		return None, 0
	if len(lefty ) == 0 or len(righty ) == 0 :
		return None, 0
	line_image = np.zeros((height, width, 3), dtype=np.uint8)
	# line_image = (np.dstack((image, image, image))*255).astype('uint8')
	window_img = np.zeros_like(line_image)


#二阶拟合，求切线
	left_line, left_fit, left_x0, kl  = line_of_poly( leftx, lefty,   int(height-height/4-1),  height -1, int( height / 6)   )  # x = ay^2 + by + c
	right_line, right_fit, right_x0, kr  = line_of_poly( rightx, righty,  int(height-height/4-1), height-1, int(height/6)  )

#中间目标线
	k0 = (kl + kr)/2
	x0 = (left_x0 + right_x0)/2
	y0 = height - 1
	y1 =  int(height/6) 
	x1 = k0 * (y1 - y0) + x0
	avg_theta = math.atan2( y1 - y0, x1-x0 )
	x0 = int(x0)
	x1 = int( x1 )
	goal_line = np.array([x0, y0, x1, y1 ])
	
	# Color in left and right line pixels
	line_image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [234, 217, 53]
	line_image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [234, 217, 53]

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
		cv2.line( line_image, (xl0,y0), (xl1, y1), (76, 177, 34), 1 )
		cv2.line( line_image, (xr0,y0), (xr1, y1), (76, 177, 34),  1  )
	# x1,y1,x2,y2 =  lines[0].reshape(4)
	# cv2.line( line_image, (x1,y1), (x2, y2), (0, 0, 255), 2 )
	# x1,y1,x2,y2 =  lines[1].reshape(4)
	# cv2.line( line_image, (x1,y1), (x2, y2), (0, 0, 255), 2 )
	x1,y1,x2,y2 =  goal_line.reshape(4)
	cv2.line( line_image, (x1,y1), (x2, y2), (36, 28, 237), 3 )

	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (29,230, 181))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (29,230, 181))
	line_image = cv2.addWeighted(line_image, 1, window_img, 0.3, 0)
	d_center =int( width/2 - (left_x0 + right_x0)/2)
	return line_image, avg_theta, d_center



def line_fit_with_contours_image( image, lane_width ):
	width = image.shape[1]
	height = image.shape[0]
	# Take a histogram of the bottom half of the image
	histogram = np.sum(image[height//2:,:], axis=0)  #height//2
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
	margin = int(width/16)
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

	noLeft = False
	noRight = False

	if lefty is None or len(lefty ) == 0:
		noLeft = True
	if righty is None or len(righty) == 0 :
		noRight = True

	if noLeft and noRight :
		return None, 0, 0, False, False

	line_image = np.zeros((height, width, 3), dtype=np.uint8)
#image to draw the move window used to detect line points
	window_img = np.zeros_like(line_image)

#used to find stop line and obstacles
	lane_img = image.copy()
	cnt = int(height / 10)
	ploty = np.linspace(0,  height -1, cnt  )


#二阶拟合，求切线
	if not noLeft :
		left_line, left_fit, left_x0, kl  = line_of_poly( leftx, lefty,   int(height-height/4-1),  height -1, int( height / 6)   )  # x = ay^2 + by + c
		# Color in left and right line pixels
		line_image[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [234, 217, 53]
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		for i in range(cnt - 1) :
			xl0 = int( left_fitx[i])
			y0 = int(ploty[i])
			xl1 = int(left_fitx[i + 1])
			y1 = int(ploty[i+1])
			cv2.line( line_image, (xl0,y0), (xl1, y1), (76, 177, 34), 1 )

		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		# Draw the lane onto the warped blank image
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (29,230, 181))
		lane_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = 0

	else:
		kl = 0  #kr
		left_x0 = 0

	if not noRight:
		right_line, right_fit, right_x0, kr  = line_of_poly( rightx, righty,  int(height-height/4-1), height-1, int(height/6)  )
		line_image[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [234, 217, 53]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		for i in range(cnt - 1) :
			xr0 = int(right_fitx[i])
			y0 = int(ploty[i])
			xr1 = int(right_fitx[i+1])
			y1 = int(ploty[i+1])
			cv2.line( line_image, (xr0,y0), (xr1, y1), (76, 177, 34),  1  )

		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (29,230, 181))
		lane_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = 0
	
	else:
		kr = kl
		right_x0 = width

	if noLeft :
		kl = kr

#中间目标线
	k0 = (kl + kr)/2
	x0 = (left_x0 + right_x0)/2
	y0 = height - 1
	y1 =  int(height/6) 
	x1 = k0 * (y1 - y0) + x0
	avg_theta = math.atan2( y1 - y0, x1-x0 )
	x0 = int(x0)
	x1 = int( x1 )
	goal_line = np.array([x0, y0, x1, y1 ])

	x1,y1,x2,y2 =  goal_line.reshape(4)
	cv2.line( line_image, (x1,y1), (x2, y2), (36, 28, 237), 3 )

	d_center =int( width/2 - (left_x0 + right_x0)/2)

	line_image = cv2.addWeighted(line_image, 1, window_img, 0.3, 0)

	stopline, obstacles =  find_stop_line_and_obstacle(line_image, lane_img, width, height,lane_width) # left_x0, right_x0,  margin )
	return line_image, avg_theta, d_center, stopline, obstacles


def find_stop_line_and_obstacle(line_image, zero_line_image, width, height, lane_width):
	image, contours, hierarchy = cv2.findContours(zero_line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE   )
	cv2.drawContours( line_image, contours, -1, (196, 0, 0), 2)
	obstacles = False
	stopline = False
	wid_cnt = 0
	y0 = height
	for i in range(0, len(contours)):
		isObt = False
		x,y,w,h = cv2.boundingRect( contours[i] )
		if  w > lane_width/3 :
			obstacles = True
			cv2.rectangle(line_image, (x,y), (x+w, y+h), (0,0,196), 2)
		if w >= lane_width  and h<35 :
			cv2.rectangle(line_image, (x,y), (x+w, y+h), (0,196,196), 2)
			wid_cnt = wid_cnt + 1
		if wid_cnt  == 4:
			stopline = True
			obstacles = False
	return stopline, obstacles


def find_stop_line_and_obstacle_old(line_image, zero_line_image, width, height, left_x0, right_x0, margin ):
	image, contours, hierarchy = cv2.findContours(zero_line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE   )
	cv2.drawContours( line_image, contours, -1, (196, 0, 0), 2)
	obstacles = False
	stopline = False
	wid_cnt = 0
	lane_width = int(right_x0 - left_x0) - 2*margin -  50  # width - 150 - 2*margin  # 
	y0 = height
	# wid_contours = []
	# obt_xl = int((left_x0 + right_x0)/2 - lane_width / 4)
	# obt_xr = int((left_x0 + right_x0)/2 + lane_width / 4)
	# print(obt_xl, obt_xr)
	for i in range(0, len(contours)):
#		area = cv2.contourArea(contours[i] )
#最小内接矩形 center(x,y) = rect[0], width rect[1][0] height rect[1][1] 旋转角度 rect[2]
		# rect = cv2.minAreaRect( contours[i])
		# box = cv2.boxPoints( rect )
		# box = np.int0(box )
		# print( rect[1][0], rect[1][1], rect[2])
		# print( box )
		# w = rect[1][0]
		# isObt = False
		# if box[0][0] >= obt_xl and box[0][0] < obt_xr:
		# 	isObt = True
		# if box[1][0] >= obt_xl and box[1][0] < obt_xr:
		# 	isObt = True
		# if box[2][0] >= obt_xl and box[2][0] < obt_xr:
		# 	isObt = True
		# if box[3][0] >= obt_xl and box[3][0] < obt_xr:
		# 	isObt = True
		# if isObt == True :
		# 	obstacles = True
		# 	cv2.drawContours(line_image, [box], 0, (0, 0,196), 2)
		# 	print('its a obstacle')
		isObt = False
		x,y,w,h = cv2.boundingRect( contours[i] )
		if  w > lane_width/3 :
			if x > left_x0 and x+w < right_x0 :
				isObt = True
			elif x < left_x0:
				if x+w > left_x0 + lane_width/4:
					isObt = True
			elif x < right_x0: 
				if right_x0 - x > lane_width/4:
					isObt = True
			if isObt == True:
				obstacles = True
				cv2.rectangle(line_image, (x,y), (x+w, y+h), (0,0,196), 2)
			#	print( x,y,w,h)

		if w >= lane_width  and h<35 :
			cv2.rectangle(line_image, (x,y), (x+w, y+h), (0,196,196), 2)
			wid_cnt = wid_cnt + 1
		if wid_cnt  == 4:
			stopline = True
			obstacles = False
	return stopline, obstacles



def center_line_fit( image, lane_width ):
	width = image.shape[1]
	height = image.shape[0]
	# Take a histogram of the bottom half of the image
	histogram = np.sum(image[height//2:,:], axis=0)  # 200
	# midpoint = np.int(histogram.shape[0]/2)
	x_base = np.argmax(histogram)
	# Choose the number of sliding windows
	nwindows = 8
	# Set height of windows
	window_height = np.int(height/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	x_current = x_base
	# Set the width of the windows +/- margin
	margin = int(width/16)
	# Set minimum number of pixels found to recenter window
	minpix = 15
	# Create empty lists to receive left and right lane pixel indices
	lane_inds = []
	
	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low =height - (window+1)*window_height
		win_y_high =height - window*window_height
		win_x_low = x_current - margin
		win_x_high = x_current + margin
		# Identify the nonzero pixels in x and y within the window
		good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
		# Append these indices to the lists
		lane_inds.append(good_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_inds) > minpix:
			x_current = np.int(np.mean(nonzerox[good_inds]))
	# Concatenate the arrays of indices
	lane_inds = np.concatenate(lane_inds)

	# Extract left and right line pixel positions
	x = nonzerox[lane_inds]
	y = nonzeroy[lane_inds]

	if y is None :
		return None, 0, 0, False, False
	if len( y ) == 0 :
		return None, 0, 0, False, False
	line_image = np.zeros((height, width, 3), dtype=np.uint8)
	window_img = np.zeros_like(line_image)
#二阶拟合，求切线
	line, fit, c_x0, kl  = line_of_poly( x, y,   int(height-height/4-1),  height -1, int( height / 6)   )  # x = ay^2 + by + c

#中间目标线
	k0 = kl
	x0 = c_x0
	y0 = height - 1
	y1 =  int(height/6) 
	x1 = k0 * (y1 - y0) + x0
	avg_theta = math.atan2( y1 - y0, x1-x0 )
	x0 = int(x0)
	x1 = int( x1 )
	goal_line = np.array([x0, y0, x1, y1 ])
	
	# Color in left and right line pixels
	line_image[nonzeroy[lane_inds], nonzerox[lane_inds]] = [234, 217, 53]

	cnt = int(height / 10)
	ploty = np.linspace(0,  height -1, cnt  )
	fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
	for i in range(cnt - 1) :
		x0 = int( fitx[i])
		y0 = int(ploty[i])
		x1 = int(fitx[i + 1])
		y1 = int(ploty[i+1])
		cv2.line( line_image, (x0,y0), (x1, y1), (76, 177, 34), 1 )
	

	line_window1 = np.array([np.transpose(np.vstack([fitx-margin, ploty]))])
	line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx+margin, ploty])))])
	line_pts = np.hstack((line_window1, line_window2))
	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([line_pts]), (29,230, 181))

	x1,y1,x2,y2 =  goal_line.reshape(4)
	cv2.line( line_image, (x1,y1), (x2, y2), (36, 28, 237), 3 )

	line_image = cv2.addWeighted(line_image, 1, window_img, 0.3, 0)

	d_center =int( width/2 - c_x0 )

	lane_img = image.copy()
	lane_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = 0
	stopline, obstacles =  find_stop_line_and_obstacle(line_image, lane_img, width, height, lane_width) # 50, width-50, margin )
	return line_image, avg_theta, d_center, stopline, obstacles
