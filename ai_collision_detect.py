import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import sys
from time import time
import os
import pickle
import threading

from web_camera import WebController, open_cam_onboard


class AICollisionDetecter():
    def __init__(self, model_file  = 'best_model.pth'):
        ''' 
        Create and publish variables needed on many of 
        the web handlers.
        '''
        print("init ai collision detecter...")
        
        model = torchvision.models.alexnet(pretrained=False)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)
        model.load_state_dict(torch.load('best_model.pth'))
      #  IncompatibleKeys(missing_keys=[], unexpected_keys=[])
        self.device = torch.device('cuda')
        self.model = model.to(self.device)
        mean = 255.0 * np.array([0.485, 0.456, 0.406])
        stdev = 255.0 * np.array([0.229, 0.224, 0.225])
        self.normalize = torchvision.transforms.Normalize(mean, stdev)
        self.req_to_quit = False
        self.blocked = False
        self.stop_line = False
        self.event = None
        self.image_timestamp  = time()
        self.image = None
        print("ai collisiont detecter initialed!")

    def detect_collision(self, image ):
        x = self.preprocess(image)
        y = self.model(x)
        # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
        y = F.softmax(y, dim=1)
        y = y.flatten()
        ny = y.detach().cpu()
        ny = ny.numpy()
        idx = np.argmax( ny )
        # print( idx, ny[idx])
        return idx, ny[idx]

    def preprocess(self, image):
        x = image
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).float()
        x = self.normalize(x)
        x = x.to(self.device)
        x = x[None, ...]
        return x

    def update_image(self, image =None):
        if image is None:
            return
        self.image = image
        self.image_timestamp  = time()
        if self.event is not None:
            self.event.set()

    def stop(self ):
        print('required to stop the detect loop...')
        self.req_to_quit = True
        if self.event is not None:
            self.event.set()
    
    def detect_loop(self ):
        print('start ai collision detect thread loop...')
        last_served_timestamp = time()
        self.event  =  threading.Event()
        while True:
            if self.req_to_quit :
                break
            if self.image is not None and  last_served_timestamp < self.image_timestamp :
                idx, perc = self.detect_collision( self.image )
                self.blocked = False
                self.stop_line = False
                if idx == 0 and perc > 0.7 :
                    self.blocked = True
                    print('blocked!', int(100*perc))
                elif idx == 2 and perc > 0.7:
                    self.stop_line = True
                    print('Stop line!', int(100*perc))
            self.event.wait()
        print( ' ai collision detect thread stoped.')


if __name__ == '__main__':
    width = 320
    height = 240
    sensor_id = 0
    flip_method = 2
    cap = open_cam_onboard(width,height, sensor_id, flip_method )
    if not cap.isOpened():
        sys.exit('failed to open camera!')

    WINDOW_NAME ="AI lane follow"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    cv2.setWindowTitle(WINDOW_NAME, 'AI lane follow')


    cal_file = 'cal120-' + str(width) + '-' + str(height) + '.p'
    with open(cal_file, 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']
    # m, m_inv,src =transform_matrix_640()
    # if width == 320:
	#     m, m_inv,src = transform_matrix_320()
    elapsed = 0
    ctrl_theta = 0
    d_center = 0
    failedCnt = 0
    detector = AICollisionDetecter()
    start_time = time()
    labels = ['block', 'free', 'stop']
    while True:
        start1 = time()
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the window
            # If yes, terminate the program
            break
        ret, image = cap.read() #grap the next image frame
        if not ret:
            key = cv2.waitKey(40)
            if key == 27: # ESC key: quit program
	            break
            continue
        undis_image = cv2.undistort(image, mtx, dist, None, mtx)
        start = time()
        idx, perc = detector.detect_collision( undis_image )
        end = time()
        label1  = '%.3f of %s ; %.3f' % (100*perc, labels[idx],  end-start)
        cv2.putText(undis_image, label1, (5, 60),  cv2.FONT_HERSHEY_PLAIN, 1, (0,196,0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, undis_image)    
        tw = 1000 * ( time() - start1)
        tw =int( 50 - tw)
        if tw <= 0 :
            tw = 1
        key = cv2.waitKey(tw)
        if key == 27: # ESC key: quit program
            break
    cap.release()
    cv2.destroyAllWindows()
