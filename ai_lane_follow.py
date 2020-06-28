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

from web_camera import WebController, open_cam_onboard


class AILaneFollower():
    def __init__(self, model_file  = 'best_steering_model_xy.pth'):
        ''' 
        Create and publish variables needed on many of 
        the web handlers.
        '''
        print("init ai follower...")
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 2)
        state_dict = torch.load(model_file )
        model.load_state_dict(state_dict)

        self.device = torch.device('cuda')
        model = model.to(self.device)
        self.model = model.eval().half()
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
        print("ai follower initialed!")
    def follow(self, image ):
        vw = self.model( self.preprocess(image)).detach().float().cpu().numpy().flatten()
        v = vw[0]
        w =vw[1]  # (0.5 - vw[1]) / 2.0
        return v, w


    def preprocess(self, image):
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]




if __name__ == '__main__':
    width = 320
    height = 240
    sensor_id = 0
    cap = open_cam_onboard(width,height, sensor_id )
    if not cap.isOpened():
        sys.exit('failed to open camera!')

    WINDOW_NAME ="AI lane follow"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    cv2.setWindowTitle(WINDOW_NAME, 'AI lane follow')


    cal_file = 'ncamera_cal' + str(width) + '-' + str(height) + '.p'
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
    follow = AILaneFollower()
    start_time = time()
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
        v,w = follow.follow( undis_image )
        end = time()
        label1  = '%02f,%03f; %02f' % (v,w, end-start)
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


    # if  len(sys.argv) < 2 :
    #     print('python ai_lane_follow.py imgfile')
    #     sys.exit('please specify a file !')

    # image_file = sys.argv[1]
    # follow = AILaneFollower()
    # image=cv2.imread(image_file)
    # start = time()
    # v,w = follow.follow( image )
    # end = time()
    # print(image_file )
    # print( end - start )
    # print(v, w )

    # image_file = 'xy_009_3087_57a24024.jpg'
    # image=cv2.imread(image_file)
    # start = time()
    # v,w = follow.follow( image )
    # end = time()
    # print(image_file )
    # print( end - start )
    # print(v, w )

    # image_file = 'cap_imgs/img06-02-100751.jpg'
    # image=cv2.imread(image_file)
    # start = time()
    # v,w = follow.follow( image )
    # end = time()
    # print(image_file )
    # print( end - start )
    # print(v, w )

    # image_file = 'cap_imgs/img06-03-114419.jpg'
    # image=cv2.imread(image_file)
    # start = time()
    # v,w = follow.follow( image )
    # end = time()
    # print(image_file )
    # print( end - start )
    # print(v, w )


    # image_file = 'xy_009_3178_42326f58.jpg'
    # image=cv2.imread(image_file)
    # start = time()
    # v,w = follow.follow( image )
    # end = time()
    # print(image_file )
    # print( end - start )
    # print(v, w )

