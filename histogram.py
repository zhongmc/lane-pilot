# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math

from zmcrobot import ZMCRobot
from lane_detector import  canny, sobel,  transform_matrix_640,transform_matrix_320, horizen_lines, line_fit_with_image

#from combined_thresh import combined_thresh, magthresh
#from line_fit import line_fit, viz2, calc_curve, final_viz

import sys
import argparse
from time import time


def parse_args():
    # Parse input arguments
    desc = 'dectect Lanes in the image'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--file', '-f',  dest='file',
                        help='the file to analys' )
    parser.add_argument('--undisort', '-ud',  dest='undisort',
                        help='image is undisorted, do not undisort it ',
                        action='store_true')
    parser.add_argument('--algs', '-a',  dest='algorithm',
                        help='algorithm to use [canny, sobel]',
                        default='canny')
    args = parser.parse_args()
    return args

def histogram_analy(  image_file, undisort, algorithm  ):
# Read camera calibration coefficients
	start = time()

	image = cv2.imread(image_file)
	height = image.shape[0]
	width = image.shape[1]
	cal_file = 'ncamera_cal' + str(width) + '-' + str(height) + '.p'
	with open(cal_file, 'rb') as f:
	# with open('camera_cal640-480.p', 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']

	if undisort :
		undis_image = image
	else :
		undis_image = cv2.undistort(image, mtx, dist, None, mtx)
	
	if algorithm == 'canny' :
		canny_image = canny( undis_image )
	elif algorithm == 'sobel':
		canny_image = sobel( undis_image) # ,  sobel_kernel=3, mag_thresh=(50, 255))
		# canny_image[ canny_image == 1] = 255

	# print( canny_image[120])
	#	print( elapsed )
	m, m_inv, src = transform_matrix_640()
	if width == 320:
		m, m_inv, src = transform_matrix_320()

	wraped_image = cv2.warpPerspective(canny_image, m, (width, height), flags=cv2.INTER_LINEAR)

	org_wraped_image =  cv2.warpPerspective(undis_image, m, (width, height), flags=cv2.INTER_LINEAR)
	
	line_image, line_theta = line_fit_with_image(wraped_image )

	# lines,  left_fit, right_fit, line_theta,  d_center   = line_fit( wraped_image )
	# line_image = np.zeros_like(image)
	# #画二阶拟合曲线(用10段线段)
	# if lines is not None:
	# 	cnt = int(height / 10)
	# 	ploty = np.linspace(0,  height -1, cnt  )
	# 	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	# 	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	# 	for i in range(cnt - 1) :
	# 		xl0 = int( left_fitx[i])
	# 		xr0 = int(right_fitx[i])
	# 		y0 = int(ploty[i])
	# 		xl1 = int(left_fitx[i + 1])
	# 		xr1 = int(right_fitx[i+1])
	# 		y1 = int(ploty[i+1])
	# 		cv2.line( line_image, (xl0,y0), (xl1, y1), (255, 255, 0), 10 )
	# 		cv2.line( line_image, (xr0,y0), (xr1, y1), (255, 255, 0),  10  )

	# 	x1,y1,x2,y2 =  lines[0].reshape(4)
	# 	cv2.line( line_image, (x1,y1), (x2, y2), (0, 0, 255), 2 )
	# 	x1,y1,x2,y2 =  lines[1].reshape(4)
	# 	cv2.line( line_image, (x1,y1), (x2, y2), (0, 0, 255), 2 )
	# 	x1,y1,x2,y2 =  lines[2].reshape(4)
	# 	cv2.line( line_image, (x1,y1), (x2, y2), (255, 0, 0), 5 )

		# x1,y1,x2,y2 =  lines[3].reshape(4)
		# cv2.line( line_image, (x1,y1), (x2, y2), (0, 255, 0), 1)

	ctrl_theta = -np.pi/2 - line_theta
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(line_image, m_inv, (width, height))
	# Combine the result with the original image
	result_image = cv2.addWeighted(undis_image, 0.8, newwarp, 0.8, 0)

	hlines = horizen_lines( wraped_image)

	elapsed = time() - start

	label = 'w: %.3f  t:%.2f' % (ctrl_theta,  elapsed*1000)
	result_image = cv2.putText(result_image, label, (30,20), 0, 0.7, (255,0,0), 2, cv2.LINE_AA)
	print( hlines )
	if  hlines >= 4 :
		label = 'Stop  !'
		result_image = cv2.putText(result_image, label, (30,50), 0, 0.7, (0,0,196), 2, cv2.LINE_AA)


	
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

	histogram = np.sum(wraped_image[200:,:], axis=0) #height//2
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


	# histogram = np.sum(wraped_image, axis=1)
	# histogram = histogram/255
	# midpoint= np.int(histogram.shape[0]/2)
	# leftx_base = np.argmax(histogram[0:midpoint])
	# rightx_base = np.argmax(histogram[midpoint: histogram.shape[0]]) + midpoint
	# mlval  = np.amax(histogram[0:midpoint])
	# mrval = np.amax(histogram[midpoint:])
	# print('y-l %d : %d y-r: %d : %d ' %( leftx_base,  mlval, rightx_base, mrval))
	# plt.subplot(236)
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

			file_name = out_image_file + '-wraped0.png'
			cv2.imwrite(file_name, org_wraped_image)

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
	args = parse_args()
	print('Called with args:')
	print(args)
	histogram_analy(args.file, args.undisort, args.algorithm)