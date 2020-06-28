# -*- coding: utf-8 -*-


import numpy as np
import pickle
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math

from zmcrobot import ZMCRobot
from lane_detector import  canny, sobel,  transform_matrix, horizen_peaks, line_fit_with_image,line_fit_with_contours_image

#from combined_thresh import combined_thresh, magthresh
#from line_fit import line_fit, viz2, calc_curve, final_viz

import sys
import argparse
from time import time

def parse_args():
	desc = 'dectect Lanes in the image'
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--file', '-f',  dest='file', help='the file to analys' )
	parser.add_argument('--undisort', '-ud',  dest='undisort', help='image is undisorted, do not undisort it ', action='store_true')
	parser.add_argument('--algs', '-a',  dest='algorithm', help='algorithm to use [canny, sobel]', default='canny')
	parser.add_argument('--lens',  dest = 'lens', help = 'camera lens used to cap the image 120 160', default=120, type=int)
	args = parser.parse_args()
	return args

def histogram_analy(  image_file, undisort, algorithm, lens  ):
	print( image_file )
	image=cv2.imread(image_file)
	start = time()
	height = image.shape[0]
	width = image.shape[1]
	cal_file = 'cal' + str(lens ) + '-' + str(width) + '-' + str(height) + '.p'
	with open(cal_file, 'rb') as f:
		save_dict = pickle.load(f)
	mtx = save_dict['mtx']
	dist = save_dict['dist']

	if undisort :
		undis_image = image
	else :
		undis_image = cv2.undistort(image, mtx, dist, None,  mtx)
		# newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0.2, (width, height))
		# print(newmtx, mtx,  roi )
		# undis_image = cv2.undistort(image, mtx, dist, None,  newmtx)
		# x,y,w,h = roi
		# undis_image = undis_image[y:y+h, x:x+w]
		# undis_image = cv2.resize(undis_image, (width, height))

	if algorithm == 'canny' :
		canny_image = canny( undis_image )
	elif algorithm == 'sobel':
		canny_image = sobel( undis_image) # ,  sobel_kernel=3, mag_thresh=(50, 255))
		# canny_image[ canny_image == 1] = 255

	m, m_inv, src, lane_width = transform_matrix(width, height, lens)
	# if width == 640:
	# 	m, m_inv, src = transform_matrix_640()
	# elif width == 320:
	# 	m, m_inv, src = transform_matrix_320()
	# elif width == 244:
	# 	m, m_inv, src = transform_matrix_244()

	wraped_image = cv2.warpPerspective(canny_image, m, (width, height), flags=cv2.INTER_LINEAR)

	org_wraped_image =  cv2.warpPerspective(undis_image, m, (width, height), flags=cv2.INTER_LINEAR)
	
	# line_image, line_theta, d_center = line_fit_with_image(wraped_image )

	line_image, line_theta, d_center , stopline, obstacles = line_fit_with_contours_image(wraped_image, lane_width )

	# image, contours, hierarchy = cv2.findContours(wraped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE   )
	# for i in range(0, len(contours)):
	# 	x,y,w,h = cv2.boundingRect( contours[i] )
	# 	cv2.rectangle(line_image, (x,y), (x+w, y+h), (0,0,196), 3)

	ctrl_theta = -np.pi/2 - line_theta
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	if line_image is not None:
		newwarp = cv2.warpPerspective(line_image, m_inv, (width, height))
		# Combine the result with the original image
		result_image = cv2.addWeighted(undis_image, 0.8, newwarp, 0.8, 0)
	else:
		result_image = undis_image

	# hpeakidxs = horizen_peaks( wraped_image, 3)
	# stopLine = False
	# if hpeakidxs is not None:
	# 	print( hpeakidxs )
	# 	if hpeakidxs[0] > height/4 and 10 < hpeakidxs[1] - hpeakidxs[0] < 30 and 35< hpeakidxs[2] - hpeakidxs[1] < 60 :
	# 		stopLine = True

	elapsed = time() - start

	label = 'w: %.3f dc:%d  t:%.2f' % (ctrl_theta,  d_center, elapsed*1000)
	result_image = cv2.putText(result_image, label, (30,20), 0, 0.7, (255,0,0), 2, cv2.LINE_AA)
	if stopline :
		label = 'Stop line !'
		result_image = cv2.putText(result_image, label, (30,50), 0, 0.7, (0,0,255), 2, cv2.LINE_AA)
	if obstacles :
		result_image = cv2.putText(result_image, 'Obstacle, stop! ', (30,50), 0, 0.7, (0,0,255), 2, cv2.LINE_AA)

#	combo_image = cv2.addWeighted(undis_image, 0.8, line_image, 1, 1)

	#the perspect wrap rectangle
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

	if line_image is not None:
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

	histogram = np.sum(wraped_image[height//2:,:], axis=0) #height//2
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
	# image=cv2.imread(args.file)
	# cv2.imshow("ttt", image)
	# cv2.waitKey(0)
	histogram_analy(args.file, args.undisort, args.algorithm, args.lens)