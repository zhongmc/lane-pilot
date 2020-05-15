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


class WebController(tornado.web.Application):
    def __init__(self):
        ''' 
        Create and publish variables needed on many of 
        the web handlers.
        '''
        print('Starting Donkey Server...')
        this_dir = os.path.dirname(os.path.realpath(__file__))
        self.static_file_path = os.path.join(this_dir, 'templates', 'static')
        
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.image_data = None
        self.image_timestamp = time.time()

        handlers = [
            (r"/", tornado.web.RedirectHandler, dict(url="/drive")),
            (r"/drive", DriveHandler),
            (r"/video",VideoHandler),
            (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": self.static_file_path}),
            ]
        settings = {'debug': True}
        super().__init__(handlers, **settings)

    def start(self, port=8887):
        ''' Start the tornado webserver. '''
        asyncio.set_event_loop(asyncio.new_event_loop())
        print(port)
        self.port = int(port)
        self.listen(self.port)
        tornado.ioloop.IOLoop.instance().start()
       
    def run(self, img_arr=None):
        if img_arr is None:
            return  self.angle, self.throttle, self.mode, self.recording
        
        r, i = cv2.imencode('.jpg', img_arr )
        if r :
            self.image_data  =  bytes(i.data)
            self.image_timestamp  = time.time()
        return self.angle, self.throttle, self.mode, self.recording

    def shutdown(self):
        pass

class DriveHandler(tornado.web.RequestHandler):
    def get(self):
        data = {}
        self.render("templates/vehicle.html", **data)
        
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


class VideoHandler(tornado.web.RequestHandler):
    '''
    Serves a MJPEG of the images posted from the vehicle. 
    '''
    # def __init__(self):
    #     self.last_served_timestamp = time.time()

    async def get(self):
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
                   'width=(int)3264, height=(int)1848, '
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
        web.run( image )
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



