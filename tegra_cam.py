# --------------------------------------------------------
# Camera sample code for Tegra X2/X1
#
# This program could capture and display video from
# IP CAM, USB webcam, or the Tegra onboard camera.
# Refer to the following blog post for how to set up
# and run the code:
#   https://jkjung-avt.github.io/tx2-camera-with-python/
#
# Written by JK Jung <jkjung13@gmail.com>
# --------------------------------------------------------


import sys
import argparse
import subprocess

import cv2
import numpy as np
import pickle
import datetime

WINDOW_NAME = 'CameraDemo'

WINDOW_NAME_UD = 'Undisort'


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
    parser.add_argument('--dist', dest='undisort',
                        help='show undisort window as well',
                        action='store_true')
    parser.add_argument('--lens', dest='lens', help='camera lens used 120/ 106 default 120', default=120, type=int)
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--sensor_id', dest='sensor_id', help='sensor id for csi camera 0/1', default=0)

    parser.add_argument('--width', dest='image_width',
                        help='image width [1920]',
                        default=1280, type=int)
    parser.add_argument('--flip', dest='flip_method', help='flip method to change image 0: normal 1:90 turn 2: 180 turn', default=0, type = int )
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=720, type=int)
    parser.add_argument('--file', dest='file_name',
                        help='save file name [capfile]',
                        default='capfile' )
    parser.add_argument('--path', dest='save_path',
                        help='save file to [path]',
                        default='cap_imgs' )

    args = parser.parse_args()
    return args


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height, sensor_id, flip_method):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc '
                   'sensor-id={} ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(sensor_id, width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc '
                   'sensor-id={} ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)3264, height=(int)2464, '
                   'format=(string)NV12, framerate=(fraction)20/1 ! '
                   'nvvidconv flip-method={} ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(sensor_id, flip_method, width, height)
        print( gst_str )
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height, undisort ):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo foimport argparser Jetson TX2/TX1')

    if undisort :
        cv2.namedWindow(WINDOW_NAME_UD, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME_UD, width, height)
        cv2.moveWindow(WINDOW_NAME_UD, 110 + width, 100)
        cv2.setWindowTitle(WINDOW_NAME_UD, 'undisort image')



def read_cam(cap, save_path, width, height, undisort, lens, file_counter = 1):
    show_help = True
    full_scrn = False
    grid_cnt = 20
    w_step = int( width/grid_cnt ) 
    h_step = int( height/grid_cnt )
    show_grid = True
    imgCnt = file_counter
    help_text = '"Esc" to Quit, "S" to Save img "H" for show /hide this Help, "G" for grid "F" to Toggle Fullscreen'
    font = cv2.FONT_HERSHEY_PLAIN

    if undisort :
        src = np.float32(
            [[0,479],
            [192,160],  #196,160
            [446,160], #443, 160
            [639, 479]])
        src = src * (width/640)
        pts = np.array( src , np.int32)
        print( pts )
        pts = pts.reshape((-1,1,2))
        print( pts )
        try:
            calfileName = "cal" + str(lens) + '-' + str(width) + "-" + str(height) + ".p"
            with open(calfileName, 'rb') as f:
                save_dict = pickle.load(f)
                mtx = save_dict['mtx']
                dist = save_dict['dist']
        except IOError as e:
            calfileName = "ncamera_cal640-480.p"
            with open(calfileName, 'rb') as f:
                save_dict = pickle.load(f)
                mtx = save_dict['mtx']
                dist = save_dict['dist']
	
    tn = datetime.datetime.now()
    fileName = save_path + '/img' + str(lens)   + '-' + str(width ) + '-' + str(height) + '-' + tn.strftime('%m-%d-%H%M%S-')

    # fileName = save_path + '/img' + str(lens) + '-' + str(width ) + '-' + str(height) + '-'
	# image = cv2.imread(image_file)

    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        _, img = cap.read() # grab the next image frame from camera
        
        if undisort :
            undis_image = cv2.undistort(img, mtx, dist, None, mtx)

        if show_help:
            cv2.putText(img, help_text, (11, 20), font,
                        1.0, (32, 32, 32), 4, cv2.LINE_AA)
            cv2.putText(img, help_text, (10, 20), font,
                        1.0, (240, 240, 240), 1, cv2.LINE_AA)
        if show_grid:
            color1 = (196,196,196)
            color2 = (255,255,255)
            for i in range( grid_cnt ):
                y = (i+1)* h_step
                x = (i+1) * w_step
                if i == int(grid_cnt/2 - 1):
                    color = color2
                else:
                    color = color1

                cv2.line( img,   (0, y), (width-1, y), color, 1 )
                cv2.line(img ,   (x, 0), (x, height-1), color, 1 )
                if undisort :
                    cv2.line( undis_image,   (0, y), (width-1, y), color, 1 )
                    cv2.line(undis_image ,   (x, 0), (x, height-1), color, 1 )
        
        if undisort :
            cv2.polylines(undis_image,[pts],True,(0,255,255))
            cv2.imshow(WINDOW_NAME_UD, undis_image)
        
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'): # toggle help message
            show_help = not show_help
        elif key == ord('G') or key == ord('g'):
            show_grid = not show_grid
        elif key == ord('F') or key == ord('f'): # toggle fullscreen
            full_scrn = not full_scrn
            if full_scrn:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)
        elif key == ord('S') or key == ord('s'): #save image
            imgName =  fileName  + str(imgCnt )  + '.jpg'

            print('save file:%s' % imgName )
            #imgName = 'test_images/calibration%s.jpg' % str(imgCnt)
            cv2.imwrite(imgName, img, [int(cv2.IMWRITE_JPEG_QUALITY),80])
            imgCnt = imgCnt +1


def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else: # by default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width,
                               args.image_height, args.sensor_id, args.flip_method )

    if not cap.isOpened():
        sys.exit('Failed to open camera!')

    open_window(args.image_width, args.image_height , args.undisort )
    read_cam(cap, args.save_path, args.image_width, args.image_height, args.undisort,args.lens  )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()