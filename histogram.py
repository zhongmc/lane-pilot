# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math

from zmcrobot import ZMCRobot
#from combined_thresh import combined_thresh
#from line_fit import line_fit, viz2, calc_curve, final_viz

import sys
from time import time

def perspective_transform(img):
	"""
	Execute perspective transform
	"""
	img_size = (img.shape[1], img.shape[0])

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

	# src = np.float32(
	# 	[[15, 719],
	# 	[467, 350],
	# 	[845, 350],
	# 	[1263, 719]])
	# dst = np.float32(
	# 	[[320, 720],
	# 	[320, 0],
	# 	[960, 0],
	# 	[960, 720]])

	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)

	warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
	return warped, m_inv

def transform_matrix_640():
	src = np.float32(
		[[0,479],
		[192,160],
		[446,160],
		[639, 479]])

	dst = np.float32(
		[[100, 479],
		[100, 0],
		[539, 0],
		[539, 479]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	return m, m_inv, src

def transform_matrix_320():
	src = np.float32(
		[[0, 239],
		[96,80],  #196,160
		[223,80], #443, 160
		[319, 239]])

	dst = np.float32(
		[[50, 239],
		[50, 0],
		[269, 0],
		[269, 239]])
	m = cv2.getPerspectiveTransform(src, dst)
	m_inv = cv2.getPerspectiveTransform(dst, src)
	return m, m_inv, src


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
		return None, None, 0, 0
	if len(lefty ) == 0 or len(righty ) == 0 :
		return None, None, 0, 0

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

def histogram_analy(  image_file ):
# Read camera calibration coefficients
	start = time()

	image = cv2.imread(image_file)
	height = image.shape[0]
	width = image.shape[1]
	cal_file = 'camera_cal' + str(width) + '-' + str(height) + '.p'
	with open(cal_file, 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']
	undis_image = cv2.undistort(image, mtx, dist, None, mtx)
	canny_image = canny( undis_image )
	#	print( elapsed )
	m, m_inv, src = transform_matrix_640()
	if width == 320:
		m, m_inv, src = transform_matrix_320()

	wraped_image = cv2.warpPerspective(canny_image, m, (width, height), flags=cv2.INTER_LINEAR)

	# lines,  target_line, line_theta,  d_center   = hough_lines( wraped_image )
	lines,  left_fit, right_fit, line_theta,  d_center   = line_fit( wraped_image )

	line_image = np.zeros_like(image)
	#画二阶拟合曲线(用10段线段)
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





	ctrl_theta = -np.pi/2 - line_theta
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(line_image, m_inv, (width, height))
	# Combine the result with the original image
	result_image = cv2.addWeighted(undis_image, 0.8, newwarp, 0.8, 0)
	elapsed = time() - start

	label = 'w: %.3f dc: %d;t:%.2f' % (ctrl_theta, d_center, elapsed*1000)
	result_image = cv2.putText(result_image, label, (30,40), 0, 0.7, (128,0,255), 2, cv2.LINE_AA)

#	combo_image = cv2.addWeighted(undis_image, 0.8, line_image, 1, 1)

	pts = np.array( src , np.int32)
	pts = pts.reshape((-1,1,2))



	plt.subplot(231)
	b,g,r = cv2.split(undis_image)  
	img2 = cv2.merge([r,g,b])  
	plt.imshow(img2)
	plt.title("undisort  img")
	
	plt.subplot(232)
	plt.imshow(canny_image, cmap ='gray')
	plt.title('canny edged %d' % int( elapsed * 1000) )

	plt.subplot(233)
	plt.imshow(wraped_image, cmap = plt.cm.gray )
	plt.title("wraped img")

	plt.subplot(234)
	b,g,r = cv2.split(line_image)  
	img2 = cv2.merge([r,g,b])  
	plt.imshow(img2)
	# plt.imshow(line_image, cmap = plt.cm.gray )
	plt.title("line img")



	plt.subplot(235)

	cv2.polylines(result_image,[pts],True,(255,255,255))

	b,g,r = cv2.split(result_image)  
	img2 = cv2.merge([r,g,b])  
	plt.imshow(img2)
	plt.title("result  img")

	plt.tight_layout()

	histogram = np.sum(wraped_image[height//2:,:], axis=0)
	histogram = histogram / 255
	mlval  = np.amax(histogram)
	plt.subplot(236)
	plt.plot(histogram)
	plt.xlim(0, width)
	plt.ylim(0, mlval)
	plt.title("X histogram")
	midpoint= np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[0:midpoint])
	rightx_base = np.argmax(histogram[midpoint: histogram.shape[0]]) + midpoint
	mlval  = np.amax(histogram[0:midpoint])
	mrval = np.amax(histogram[midpoint:])
	print('x-l %d : %d x-r: %d : %d ' %( leftx_base,  mlval, rightx_base, mrval))


	# #histogram = np.sum(wraped[wraped.shape[0]//2:,:], axis=1)
	# histogram = np.sum(wraped_image, axis=1)
	# histogram = histogram/255
	# midpoint= np.int(histogram.shape[0]/2)
	# leftx_base = np.argmax(histogram[0:midpoint])
	# rightx_base = np.argmax(histogram[midpoint: histogram.shape[0]]) + midpoint
	# mlval  = np.amax(histogram[0:midpoint])
	# mrval = np.amax(histogram[midpoint:])
	# print('y-l %d : %d y-r: %d : %d ' %( leftx_base,  mlval, rightx_base, mrval))
	# plt.subplot(326)
	# plt.plot(histogram)
	# plt.xlim(0, height)
	# plt.ylim(0, mlval)
	# plt.title("Y histogram")
	plt.show()

	cv2.imshow('result', result_image )
    # cv2.putText(img, help_text, (11, 20), font,
    #      1.0, (32, 32, 32), 4, cv2.LINE_AA)
 	
	out_path = './out_images/'
	out_image_file = os.path.basename( image_file )
	print( out_image_file  )
	out_image_file = out_path + out_image_file.split('.')[0]

	while True:
		key = cv2.waitKey(0)
		if key == 27: # ESC key: quit program
			break
		elif key == ord('s') or key == ord('S'): # toggle fullscreen
			file_name = out_image_file + '-undisort.png'
			print('save : ', file_name  )
			cv2.polylines(undis_image,[pts],True,(0,255,255))
			cv2.imwrite(file_name, undis_image)

			file_name = out_image_file + '-canny.png'
			print('save : ', file_name  )
			cv2.imwrite(file_name, canny_image)

			file_name = out_image_file + '-wraped.png'
			print('save : ', file_name  )
			cv2.imwrite(file_name, wraped_image)

			file_name = out_image_file + '-lined.png'
			print('save : ', file_name  )
			cv2.imwrite(file_name, line_image)

			file_name = out_image_file + '-result.png'
			print('save : ', file_name  )
			cv2.imwrite(file_name, result_image)

			# file_name = out_image_file + '-plt.png'
			# plt.savefig(file_name )

	cv2.destroyAllWindows()


if __name__ == '__main__':
	if  len(sys.argv) < 2:
		print('python histogram.py imgfile')
	else:
		histogram_analy( sys.argv[1])

