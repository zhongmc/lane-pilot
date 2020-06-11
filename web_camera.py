import os
import tornado.web
import tornado.ioloop
from tornado.web import RequestHandler
from tornado.options import define, options, parse_command_line
import cv2
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import argparse

import asyncio
from threading import Thread
import sys
#from time import time
import time
import json


class WebController(tornado.web.Application):
    def __init__(self, port = 8887):
        ''' 
        Create and publish variables needed on many of 
        the web handlers.
        '''
        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')
        self.port = port
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.image_data = None
        self.cap_image_data = None
        self.pilotOn = False
        self.image_timestamp = time.time()
        self.pilot = None
        self.record_vw = False

        handlers = [
            (r"/", tornado.web.RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveHandler),
            (r"/video",VideoHandler),
            (r"/capture", CaptureHandler),
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": self.static_file_path}),
            ]
        settings = {'debug': True}
        super().__init__(handlers, **settings)

    def start(self ):
        ''' Start the tornado webserver. '''
        asyncio.set_event_loop(asyncio.new_event_loop())
        print('Starting Robot Web control Server...')
        print('Listen on: ',  self.port)
        self.listen(self.port)
        tornado.ioloop.IOLoop.instance().start()
       
    def update_image(self, image =None):
        if image is None:
            return  self.angle, self.throttle, self.mode, self.recording
        
        r, i = cv2.imencode('.jpg', image )
        if r :
            self.image_data  =  bytes(i.data)
            self.image_timestamp  = time.time()
        return self.angle, self.throttle, self.mode, self.recording

    def set_cap_image( self, image = None):
        if image is None:
            return
        r, i = cv2.imencode('.jpg', image )
        if r :
            self.cap_image_data  =  bytes(i.data)

    def capture_image( self ):
        if self.pilot is None:
            return
        img = self.pilot.capture_image()
        self.cap_image_data = None
        self.set_cap_image( img )

    def capture_collision_image(self, img_type):
        ret = self.pilot.capture_collision_image( img_type )
        return ret

    def updateDrive(self ):
        if self.pilot is None:
            return None
        ret = self.pilot.drive_car( -self.angle, self.throttle , self.pilotOn, self.recording, self.record_vw ) 
        return ret


    def shutdown(self):
        pass

class DriveHandler(tornado.web.RequestHandler):
    def get(self):
        data = {}
        self.render("templates/lane_pilot.html", **data)
        
    def post(self):
        '''
        Receive post requests as user changes the angle
        and throttle of the vehicle on a the index webpage
        '''
        data = tornado.escape.json_decode(self.request.body)
        self.application.angle = data['angle']
        self.application.throttle = data['throttle']
        self.application.mode = data['drive_mode']
        self.application.recording = data['recording']
        self.application.pilotOn = data['pilotOn']
        self.application.record_vw = data['record_vw']
#        print( self.application.angle,  self.application.throttle, self.application.mode, self.application.pilotOn )
        ret =  self.application.updateDrive()
        self.set_header('Content-Type', 'application/json; charset=UTF-8')
        self.write(json.dumps( ret ))
        self.finish()

class CaptureHandler(tornado.web.RequestHandler):
    def get(self):
        arg = self.get_query_argument('action', '')
        if arg == 'img':
            if self.application.cap_image_data  is None:
                self.send_error(404)
                return
            self.set_header("Context-Type", "image/jpg")
            self.set_header("Content-Disposition", 'filename="capturedImage.jpg"')
            self.write(self.application.cap_image_data)
            self.finish()

        elif arg == 'cap_img':
            ret = self.application.capture_image()
            self.set_header('Content-Type', 'text/html; charset=UTF-8')
            content = '<img src="/capture?action=img&id=%d" alt="cap image"></img>' % int(time.time()*1000)
            self.write(content )
            self.finish()
        elif arg == 'plain_img' or arg == 'obstacle_img':
            ret = self.application.capture_collision_image( arg )
            self.set_header('Content-Type', 'text/html; charset=UTF-8')
            content = "OK!"
            self.write(content )
            self.finish()

class VideoHandler(tornado.web.RequestHandler):
    '''
    Serves a MJPEG of the images posted from the vehicle. 
    '''
    # def __init__(self):
    #     self.last_served_timestamp = time.time()

    async def get(self):
        if self.application.image_data is None:
            self.send_error(404)
            return

        last_served_timestamp = time.time()
        while True:
            if last_served_timestamp < self.application.image_timestamp :
                last_served_timestamp =  self.application.image_timestamp
                self.set_header("Content-type", "multipart/x-mixed-replace;boundary=--boundarydonotcross")
                my_boundary = "--boundarydonotcross\n"
                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(self.application.image_data)) 
                self.write(self.application.image_data)

                try:
                    await self.flush()
                except tornado.iostream.StreamClosedError:
                    pass

            else:
                await tornado.gen.sleep( 0.1)                

WINDOW_NAME = 'CameraDemo'


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson TX2/TX1')



def open_cam_onboard(width, height, sensor_id):
	gst_str = ('nvarguscamerasrc '
                   'sensor-id={} ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)3264, height=(int)2464, '
                   'format=(string)NV12, framerate=(fraction)20/1 ! '
                   'nvvidconv flip-method=0 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(sensor_id, width, height)
	print( gst_str )
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


if __name__ == '__main__':
    cap = open_cam_onboard(320, 240, 0)
    if not cap.isOpened() :
        sys.exit('Failed to open camera!')
    web = WebController()
    
    t = Thread(target=web.start, args=())
    t.daemon = True
    t.start()
  
    # open_window(320,340)
 
    while True:
        ret,  image = cap.read()
        web.update_image( image )
        # cv2.imshow(WINDOW_NAME, image )
        key = cv2.waitKey(10)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


    
#        cv2.imshow("test", image)
#        key = cv2.waitKey(10)
#        if key == 27: # ESC key: quit program
#            break



