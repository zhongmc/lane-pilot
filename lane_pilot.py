# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import argparse
import datetime
from uuid import uuid1

from zmcrobot import ZMCRobot
from web_camera import WebController, open_cam_onboard
from ai_collision_detect import AICollisionDetecter

from ai_lane_follow import AILaneFollower

#from combined_thresh import combined_thresh, magthresh
#from line_fit import line_fit, viz2, calc_curve, final_viz

from lane_detector import  canny, sobel,  transform_matrix, horizen_peaks, line_fit_with_contours_image, center_line_fit
import sys
from time import time
from threading import Thread
import signal

class LanePilot():
	def __init__(self):
		self.angle = 0.0
		self.throttle = 0.0
		self.image_data = None
		self.pilotOn = False
		self.drive_mode = 'user'
		self.robot = None
		self.web = None
		self.capCnt = 1
		self.onTurnBack = False
		self.width = 320
		self.height = 240
		self.recording = False
		self.videoWriter = None
		self.start_time = time()
		self.frame_cnt = 0
		self.req_quit = False
		self.recording_vw = False
		self.aifollower = None
		#undisort mattrix
		self.mtx = None
		self.dist = None
		#perspect wrap 
		self.m = None
		self.m_inv = None
		self.src = None
		self.failedCnt = 0
		self.lens = 120
		self.lane_width = 100

	def start(self, width, height, sensor_id, port, serial_port, algorithm, flip_method, drive_v, lens):
		''' Start the lane pilot server  '''
		self.width = width
		self.height = height
		self.lens = lens
		cal_file = 'cal' + str(lens)+ '-' + str(width) + '-' + str(height) + '.p'
		print( 'load camera calibrate file: ', cal_file )
		with open(cal_file, 'rb') as f:
			save_dict = pickle.load(f)
		self.mtx = save_dict['mtx']
		self.dist = save_dict['dist']
		print( 'get transform matrix: ', width, height )
		self.m, self.m_inv, self.src,self.lane_width  =transform_matrix(width, height, lens)

		print('load ai follower...')
		self.aifollower = AILaneFollower()

		print('Init zmc robot at:', serial_port )
		self.robot = ZMCRobot(serial_port )

		print('start the web server...')
		self.web = WebController( port  )
		self.aicollision = AICollisionDetecter()
		self.web.pilot = self
		# self.web.start( port )
		self.mkdir('cap_imgs')
		self.mkdir('dataset')
		self.mkdir('dataset/blocked')
		self.mkdir('dataset/free')
		self.mkdir('dataset_vw')


		cap = open_cam_onboard(width,height, sensor_id, flip_method )
		if not cap.isOpened():
			print( 'failed to open camera!')
			return
		
		self.line_pilot( cap,  width, height, algorithm, drive_v )
		cap.release()
		cv2.destroyAllWindows()

	def mkdir(self, pathname ):
		try:
			print('try to mkdir:', pathname)
			os.makedirs(pathname )
		except FileExistsError:
			print('dir exist!')

	def drive_car(self, data):
		ret = {}
		ret['x'] = self.robot.x
		ret['y'] = self.robot.y
		ret['theta'] = self.robot.theta
		ret['v'] = self.robot.v
		angle = data['angle']
		throttle = data['throttle']
		self.drive_mode = data['drive_mode']
		recording = data['recording']
		pilotOn = data['pilotOn']
		record_vw = data['record_vw']
		if recording == True:
			self.startRecording()
		else:
			self.stopRecording()
		if record_vw == True:
			self.startRecordVW()
		else:
			self.stopRecordVW()
		if self.pilotOn == True and self.onTurnBack : #pilot mode and on stop line;
			return ret
		if pilotOn == False and self.pilotOn == True :
			print('stop line pilot! ')
		if self.pilotOn == False and pilotOn == True:
			print('start line polot...')
		self.pilotOn = pilotOn
		if self.pilotOn == False:
			self.robot.drive_car(throttle, angle  )
			if self.recording_vw :
				self.capture_vw_image( throttle,angle )

		# self.onTurnBack = False  # stop current turn back ???
		return ret

	def startRecording(self ):
		if self.recording == True:
			return

		self.start_time = time()
		self.frame_cnt = 0
		self.recording = True
		tn = datetime.datetime.now()
		filename = 'cap_videos/video' + tn.strftime('%m-%d-%H%M%S.avi')
		print('cap video to: ', filename )
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.videoWriter = cv2.VideoWriter(filename, fourcc, 20, (self.width, self.height ) )
		if self.videoWriter is None:
			print( 'failed to open video writer !')

	def stopRecording(self ):
		if self.recording == False:
			return
		self.recording = False
		self.videoWriter.release()

	def startRecordVW(self ):
		if self.recording_vw == True:
			return
		self.recording_vw = True
		print('start to record vw imgs to dataset_vw')

	def stopRecordVW(self):
		if self.recording_vw == False:
			return
		self.recording_vw = False
		print('stop record vw imgs.')

#to do capture a large image???
	def capture_image(self):
		tn = datetime.datetime.now()
		filename = 'cap_imgs/img' + str(self.lens)  + tn.strftime('%m-%d-%H%M%S.jpg')
		# fileName = self.capFile + str( self.capCnt ) + '.jpg'
		self.capCnt = self.capCnt + 1
		cv2.imwrite(filename, self.image_data, [int(cv2.IMWRITE_JPEG_QUALITY),80])
		print('capture img:', filename)
		return self.image_data


	def capture_collision_image(self, img_type ):
		path = 'dataset/blocked/'
		if img_type == 'plain_img':
			path = 'dataset/free/'
		tn = datetime.datetime.now()
		filename = path  + tn.strftime('%m-%d-%H%M%S.jpg')
		cv2.imwrite(filename, self.image_data, [int(cv2.IMWRITE_JPEG_QUALITY),80])
		print('capture img:', filename)
		return True

	def capture_vw_image(self, v, w ):

		if self.onTurnBack:
			return		
		if self.frame_cnt%4 == 0:
			astr = str(uuid1())
			astr = astr[0:8]
			filename = 'dataset_vw/xy_%03d_%04d_%s.jpg' % (int(100*v), int(1000*w + 3000), astr )
			cv2.imwrite(filename, self.image_data, [int(cv2.IMWRITE_JPEG_QUALITY),80])

	def signal_handler(self, signal, frame):
		print('ctrl + c pressed ...')
		str = input("[y/n] to quit or not?")
		if str == 'y' or str == 'Y':
			self.req_quit = True

	def do_vision_pilot(self, undis_image, algorithm, drive_v ):
		if algorithm == 'sobel':
			canny_image = sobel( undis_image )
		else:
			canny_image = canny( undis_image )
			#canny_image = combined_thresh(undis_image)
		wraped_image = cv2.warpPerspective(canny_image, self.m, (self.width, self.height))
		stopline = False
		obstacles = False
		# drive_v = 0.10
		try:
			# line_image, line_theta, d_center = line_fit_with_image(wraped_image )
			if self.drive_mode == 'c-line':
				line_image, line_theta, d_center , stopline, obstacles  = center_line_fit( wraped_image, self.lane_width )
			else:
				line_image, line_theta, d_center , stopline, obstacles  = line_fit_with_contours_image( wraped_image,self.lane_width )
		except BaseException as e:
			#print('line detect failed ' )
			line_image = None
			self.failedCnt = self.failedCnt + 1
				# lines,  left_fit, right_fit, line_theta,  d_center   = line_fit( wraped_image )
				# line_image = np.zeros_like(image)
		if self.onTurnBack :
			label = 'turn back...'
			if self.robot.turn_back_ok():
				self.onTurnBack = False
				label = 'turn back OK'
		else:
			if  line_image is not None:
				ctrl_theta = np.pi/2 + line_theta
				w = -1.8*ctrl_theta + 5 * d_center/self.width
				if self.pilotOn :
					self.robot.drive_car(drive_v, w )
					if self.recording_vw :
						self.capture_vw_image( drive_v, w )
				label = 'd:%d w:%.3f' % (d_center,w)
				self.failedCnt = 0
			else:
				if self.pilotOn and self.failedCnt > 5:
					print('failed dectect...')
					self.robot.drive_car(0, 0 ) # stop the car when none line 
					# if self.recording_vw :
					# 	self.capture_vw_image( 0, 0 )
				label = 'failed to detect'
				self.failedCnt = self.failedCnt + 1
				# if obstacles == True:
				# 	self.robot.drive_car(0, 0)
				# 	if self.recording_vw :
				# 		self.capture_vw_image( 0, 0 )
				# 	label = "Obstacle!"					
			if stopline == True and self.pilotOn == True : #stop lines turn arround
				self.onTurnBack = True
				self.robot.turn_back()
				# if self.recording_vw :
				# 	self.capture_vw_image( 0.0, 0.2 )
				self.capture_image()
				label = 'turn back...'
				print('stop and turn back.')
			# Warp the blank back to original image space using inverse perspective matrix (Minv)
			if line_image is not None:			
				newwarp = cv2.warpPerspective(line_image, self.m_inv, (self.width, self.height))
		# Combine the result with the original image
				result_image = cv2.addWeighted(undis_image, 1, newwarp, 0.5, 0)
			else:
				result_image = undis_image.copy()
			cv2.putText(result_image, label, (30,40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,40), 2, cv2.LINE_AA)
			return result_image



	def line_pilot(self, cap, width, height, algorithm, drive_v ):
		# ncamera_cal for 3264*2464
		# camera_cal  for 3264*1848
		t = Thread(target=self.web.start, args=())
		t.daemon = True
		t.start()
  
		t1 = Thread( target = self.aicollision.detect_loop, args=())
		t1.start()

  #catch the Ctrl-C
		signal.signal(signal.SIGINT, self.signal_handler )
		elapsed = 0

		self.start_time = time()
		while True:
			start = time()
			if self.req_quit == True:
				print( "required to quit!")
				break

			ret, image = cap.read() #grap the next image frame
			if not ret:
				key = cv2.waitKey(40)
				if key == 27: # ESC key: quit program
					break
				continue
			undis_image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
			self.image_data = undis_image

			if (self.frame_cnt % 2) == 0:
				self.aicollision.update_image( undis_image )

			if self.drive_mode == 'learn':
				v, w = self.aifollower.follow( undis_image )
				result_image = undis_image
				if self.pilotOn :
					self.robot.drive_car(0.09, w )
				label = 'v:%.3f,w:%.3f' % (v, w)
				cv2.putText(result_image, label, (30,40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,40), 2, cv2.LINE_AA)
			else:
				result_image = self.do_vision_pilot( undis_image,algorithm, drive_v )

			elapsed = int( time() - self.start_time)
			elapsed_label = '%02d:%02d %03d'  % (int(elapsed/60), elapsed%60, self.frame_cnt )	
			self.frame_cnt+=1
			cv2.putText(result_image, elapsed_label, (5, 15),  cv2.FONT_HERSHEY_PLAIN, 1, (0,196,0), 2, cv2.LINE_AA)
			v_label = '%.2f' % (self.robot.v)
			cv2.putText(result_image, v_label, (width-50, 15),  cv2.FONT_HERSHEY_PLAIN, 1, (196,196,196), 2, cv2.LINE_AA)

			self.web.update_image( result_image )
		
			if self.recording == True and  self.videoWriter is not None:
				self.videoWriter.write( result_image )
#		cv2.imshow(WINDOW_NAME, result_image )
			tw = 1000 * ( time() - start)
			tw =int( 50 - tw)
			if tw <= 0 :
			 	tw = 1
			key = cv2.waitKey(tw)
			if key == 27: # ESC key: quit program
				break
		self.robot.shutdown()
		self.aicollision.stop()


def parse_args():
	# Parse input arguments
	desc = 'Lane pilot robot'
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
                        help='image width [320]',
                        default=320, type=int)
	parser.add_argument('--height', dest='image_height',
                        help='image height [240]',
                        default=240, type=int)
	parser.add_argument('--port', dest='port',
                        help='web ctrl listen port [8887]',
                        default=8887, type=int)
	parser.add_argument('--serial', dest='serial_port', help='robot serial port [/dev/ttyACM0]', default='/dev/ttyACM0' )
	parser.add_argument('--algs', dest='algorithm', help='algorithm used to detect the line[canny,  sobel]',  default='canny')
	parser.add_argument('--flip', dest = 'flip_method', help='flip method used to flip the img 0:normal 1:90 2:up down 180', default = 0, type = int)
	parser.add_argument('--vilocity', '-v', dest = 'drive_v', help='vilocity of the robot,default 0.08', default = 0.08, type=float) 

	parser.add_argument('--lens', dest = 'lens', help='the camera lens 120 / 160', default=160, type=int )
	args = parser.parse_args( )
	return args

def main():
	args = parse_args()
	print('Called with args:')
	print(args)
	print('OpenCV version: {}'.format(cv2.__version__))
	pilot = LanePilot()
	pilot.start(args.image_width, args.image_height, args.sensor_id, args.port, args.serial_port, args.algorithm, args.flip_method, args.drive_v, args.lens)
	
if __name__ == '__main__':
	main()

