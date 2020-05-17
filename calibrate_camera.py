import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import sys
import argparse



def parse_args():
    # Parse input arguments
	desc = 'calibrate camera with chess board images'
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--imgpath', dest='imgpath', help='chess board img path', default='camera_cal/')
	parser.add_argument('--imgfile', dest='imgfile',  help='the image file name etc[calibration.jpg]', default='calibration.jpg' )
	parser.add_argument('--outfile', dest='outfile', help='the file save the result to', default='calibrate_camera.p')
	args = parser.parse_args()
	return args




def calibrate_camera( filePath, fileName ):
	# Mapping each calibration image to number of checkerboard corners
	# Everything is (9,6) for now
	objp_dict = {
		1: (9, 6),
		2: (9, 6),
		3: (9, 6),
		4: (9, 6),
		5: (9, 6),
		6: (9, 6),
		7: (9, 6),
		8: (9, 6),
		9: (9, 6),
		10: (9, 6),
		11: (9, 6),
		12: (9, 6),
		13: (9, 6),
		14: (9, 6),
		15: (9, 6),
		16: (9, 6),
		17: (9, 6),
		18: (9, 6),
		19: (9, 6),
		20: (9, 6),
	}

	# List of object points and corners for calibration
	objp_list = []
	corners_list = []

	file = fileName.split('.')[0]
	fileex  = '.' +   fileName.split('.')[1]
	# Go through all images and find corners
	for k in objp_dict:
		nx, ny = objp_dict[k]

		# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((nx*ny,3), np.float32)
		objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

		# Make a list of calibration images
		fname =   filePath +'/' + file + str(k) + fileex   #'camera_cal/calibration%s.jpg' % str(k)
		img = cv2.imread(fname)

		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

		# If found, save & draw corners
		if ret == True:
			# Save object points and corresponding corners
			objp_list.append(objp)
			corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
				criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
			if corners2.any():
				corners_list.append(corners2)
			else:
				corners_list.append(corners)
			#corners_list.append(corners)

			# Draw and display the corners
			#cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
			#plt.imshow(img)
			#plt.show()
			#print('Found corners for %s' % fname)
		else:
			print('Warning: ret = %s for %s' % (ret, fname))

	# Calibrate camera and undistort a test image
#	img = cv2.imread('test_images/straight_lines1.jpg')

	img_size = (img.shape[1], img.shape[0])
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objp_list, corners_list, img_size,None,None)
	return mtx, dist


if __name__ == '__main__':
	args = parse_args()
	print('Called with args:')
	print(args)
	mtx, dist = calibrate_camera(args.imgpath, args.imgfile )

	save_dict = {'mtx': mtx, 'dist': dist}
	with open(args.outfile, 'wb') as f:
		pickle.dump(save_dict, f)

	# Undistort example calibration image

	file = args.imgfile.split('.')[0]
	fileex  = '.' +   args.imgfile.split('.')[1]
	imgname = args.imgpath + file + str(5) + fileex
	img = mpimg.imread( imgname )
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	plt.imshow(dst)
	plt.savefig('undistort_calibration.png')

	cv2.imshow("orignal", img)
	cv2.imshow("undisort", dst)
	cv2.waitKey(0)