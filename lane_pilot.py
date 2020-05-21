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
from web_camera import WebController, open_cam_onboard
#from combined_thresh import combined_thresh
#from line_fit import line_fit, viz2, calc_curve, final_viz

from lane_detector import line_fit, canny, draw_lines, transform_matrix_640,transform_matrix_320
import sys
from time import time
from threading import Thread

class LanePilot():
	def __init__(self):
		self.angle = 0.0
		self.throttle = 0.0
		self.image_data = None
		self.pilotOn = False
		self.robot = None
		self.web = None

	def start(self, width, height, sensor_id, port=8887):
		''' Start the lane pilot server  '''
		self.robot = ZMCRobot()
		self.web = WebController()

		self.web.pilot = self
		# self.web.start( port )
		
		cap = open_cam_onboard(width,height, sensor_id )
		if not cap.isOpened():
			print( 'failed to open camera!')
			return
		
		self.line_pilot( cap,  width, height )
		cap.release()
		cv2.destroyAllWindows()

	def drive_car(self, angle, throttle, pilot ):
		self.pilotOn = pilot
		if self.pilotOn == False:
			self.robot.drive_car(throttle, angle  )
		ret = {}
		ret['x'] = self.robot.x
		ret['y'] = self.robot.y
		ret['theta'] = self.robot.theta
		return ret

	def line_pilot(self, cap, width, height ):
		cal_file = 'camera_cal' + str(width) + '-' + str(height) + '.p'
		with open(cal_file, 'rb') as f:
			save_dict = pickle.load(f)
		mtx = save_dict['mtx']
		dist = save_dict['dist']
		t = Thread(target=self.web.start, args=())
		t.daemon = True
		t.start()
  
		m, m_inv,src =transform_matrix_640()
		if width == 320:
			m, m_inv,src = transform_matrix_320()
		elapsed = 0
		ctrl_theta = 0
		d_center = 0
		while True:
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
				draw_lines(line_image, lines, left_fit, right_fit, height )
				ctrl_theta = np.pi/2 + line_theta
				if self.pilotOn :
					self.robot.drive_car(0.06, ctrl_theta )
			else:
				if self.pilotOn:
					self.robot.drive_car(0, 0 ) # stop the car when none line 
		# Warp the blank back to original image space using inverse perspective matrix (Minv)
			newwarp = cv2.warpPerspective(line_image, m_inv, (width, height))
	# Combine the result with the original image
			result_image = cv2.addWeighted(undis_image, 1, newwarp, 0.5, 0)
			label = 'w: %.3f dc: %d;t:%.2f' % (ctrl_theta, d_center, elapsed*1000)
			result_image = cv2.putText(result_image, label, (30,40), 0, 0.5, (128,0,128), 2, cv2.LINE_AA)
			self.web.update_image( result_image )
			elapsed = time() - start
#		cv2.imshow(WINDOW_NAME, result_image )
			key = cv2.waitKey(10)
			if key == 27: # ESC key: quit program
				break
		self.robot.shutdown()



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
                        default=320, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=240, type=int)
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
	pilot = LanePilot()
	pilot.start(args.image_width, args.image_height, args.sensor_id)

if __name__ == '__main__':
	main()

