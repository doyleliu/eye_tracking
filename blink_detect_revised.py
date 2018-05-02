from scipy.spatial import distance as dist
from imutils.video import FileVideoStream, VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys, os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

from PIL import Image
import datasets, hopenet, utils
from skimage import io

import math

from queue import Queue

import unittest
import threedface.api as api


def parse_args():
    """input arguments"""
    parser = argparse.ArgumentParser(description='using to test driver drowiness.')
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='hopenet_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='mmod_human_face_detector.dat', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video',
          default='test.mp4')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file', default='output.mp4')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = parser.parse_args()
    return args


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    # eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


def RaySphereIntersect(rayOrigin, rayDir, sphereOrigin, sphereRadius):
    # eye gaze detection
    dx = rayDir[0]
    dy = rayDir[1]
    dz = rayDir[2]
    x0 = rayOrigin[0]
    y0 = rayOrigin[1]
    z0 = rayOrigin[2]
    cx = sphereOrigin[0]
    cy = sphereOrigin[1]
    cz = sphereOrigin[2]
    r = sphereRadius

    a = dx*dx + dy*dy + dz*dz
    b = 2*dx*(x0-cx) + 2*dy*(y0-cy) + 2*dz*(z0-cz)
    c = cx*cx + cy*cy + cz*cz + x0*x0 + y0*y0 + z0*z0 + -2*(cx*x0 + cy*y0 + cz*z0) - r*r

    disc = b*b - 4*a*c

    t = (-b - math.sqrt(b*b - 4*a*c))/2*a

    # This implies that the lines did not intersect, point straight ahead
    if (b*b - 4 * a*c < 0):
        return [0, 0, -1]
    else:
        return rayOrigin + rayDir * t


def get_pupil_location(eye):
    A = eye[1] + eye[2] + eye[4] + eye[5]
    B = A / 4

    return B

def GetPupilPosition(eye):
    eye_transpose = np.transpose(eye)
    print(eye_transpose.shape)
    irisLdmks3d = np.array([np.mean(eye_transpose[0]), np.mean(eye_transpose[1]), np.mean(eye_transpose[2])])
    
    return irisLdmks3d


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def rotationMatrixToEulerAngles(R) :

    # assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def estimate_gaze(headpos, eye, threedfaciallandmark, isleft_eye):
    eulerAngles = headpos
    rotMat = eulerAnglesToRotationMatrix(eulerAngles)
    # eyeLdmks3d = eye
    if(isleft_eye):
        eyeLdmks3d = threedfaciallandmark[36:42]
    else:
        eyeLdmks3d = threedfaciallandmark[42:48]

    pupil = GetPupilPosition(eyeLdmks3d)
    # pupil = get_pupil_location(eyeLdmks3d)
    rayDir = pupil / cv2.norm(pupil)

    # cv::Mat faceLdmks3d = clnf_model.GetShape(fx, fy, cx, cy);
    # faceLdmks3d = faceLdmks3d.t();

    offset = [0, -3.5, 7.0]
    # offset = [7.0, -3.5, 0]
    offset = np.transpose(offset)
    # print(rotMat.shape)
    # print(offset.shape)
    # ceshi = np.dot(rotMat, offset)
    # print(threedfaciallandmark[36].shape)
    # print(ceshi.shape)
    if(isleft_eye):
        eyeballCentre = (threedfaciallandmark[36] + threedfaciallandmark[39]) / 2.0 + np.transpose(np.dot(rotMat, offset)) 
    else:
        eyeballCentre = (threedfaciallandmark[42] + threedfaciallandmark[45]) / 2.0 + np.transpose(np.dot(rotMat, offset)) 

    gazeVecAxis = RaySphereIntersect([0, 0, 0], rayDir, eyeballCentre, 12) - eyeballCentre
    gaze_absolute = gazeVecAxis / cv2.norm(gazeVecAxis)

    # gaze_absolute = rotationMatrixToEulerAngles(gaze_absolute)
    return gaze_absolute


def judge_fatigue(perclos, dist):
    # detect the driver fatigue using the input factors with perclos and eye gaze distraction
    if(perclos > 0.5):
        print("false")
        return False
    elif(perclos > 0.3 and dist == False):
        print("false")
        return False
    else:
        print("true")
        return True


def distract_detect(eyegazequeue):
    cnt = 0
    final_queue = Queue()
    for i in range(0, eyegazequeue.qsize()):
        tmp_queue = eyegazequeue.get()
        final_queue.put(tmp_queue)
        if(tmp_queue is False):
            cnt = cnt + 1
        
    if(cnt > final_queue.qsize() / 3):
        return final_queue, False
    else:
        return final_queue,True

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    batch_size = 1
    # gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_path = args.video_path

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path , map_location='cpu')
    model.load_state_dict(saved_state_dict)

    print('Loading data.')
    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    print('Ready to test network.')

    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)


# initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # print(lStart, lEnd)
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    vs = FileVideoStream(video_path).start()

    fileStream = True
    time.sleep(1.0)
    # euler angle
    yaw_predicted = 0
    pitch_predicted = 0
    roll_predicted = 0
    # count numbers
    frame_cnt = 0
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    bbox_height = 0

    video_dir = 'output/output.avi'
    fps = args.fps
    img_size = (360, 202)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, args.fps, (480, 202))

    # frame_per_30 = 0
    frame_per_count = 0
    # frame queue
    frame_queue = Queue()
    # eye gaze distraction
    distract = Queue()
    # 3d face alignment detection
    fa = api.FaceAlignment(api.LandmarksType._3D, enable_cuda=False)
    preds = []

    while True:
        # leftEyeHull = 0；
        
        frame_cnt = frame_cnt + 1
        if fileStream and not vs.more():
            break
        # if(frame_cnt % 2 == 0):
        #     continue
        frame = vs.read()
        frame = imutils.resize(frame, width=360)
        # change the state
        frame = cv2.flip(frame, -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        headpos = []

        # if(frame_cnt % 5 == 0):
        if True:
            cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = cnn_face_detector(cv2_frame, 1)
            # print("here")
            # print(dets)
            for idx, det in enumerate(dets):
                x_min = det.rect.left()
                y_min = det.rect.top()
                x_max = det.rect.right()
                y_max = det.rect.bottom()
                conf = det.confidence

                if(conf > 1.0):
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)
                    x_min -= 2 * bbox_width / 4
                    x_max += 2 * bbox_width / 4
                    y_min -= 3 * bbox_height / 4
                    y_max += bbox_height / 4
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(frame.shape[1], x_max)
                    y_max = min(frame.shape[0], y_max) 
                    y_max = int(y_max)
                    y_min = int(y_min)
                    x_max = int(x_max)
                    x_min = int(x_min)

                    img = cv2_frame[y_min:y_max,x_min:x_max]
                    img = Image.fromarray(img)

                    # transform
                    img = transformations(img)
                    img_shape = img.size()
                    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                    img = Variable(img)

                    yaw, pitch, roll = model(img)
                    yaw_predicted = F.softmax(yaw)
                    pitch_predicted = F.softmax(pitch)
                    roll_predicted = F.softmax(roll)

                    # Get continuous predictions in degrees.
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
                    # print("first:")
                    # print([yaw_predicted, pitch_predicted, roll_predicted])

                    # Print new frame with cube and axis
                    # txt_out.write(str(frame_cnt) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                    if(frame_cnt > 1):
                        utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)

        if(frame_cnt > 1):
            utils.draw_axis(frame,  yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)

        headpos = [yaw_predicted, pitch_predicted, roll_predicted]

        tmpframe = cv2.imwrite("tmp.jpg",frame)
        input = io.imread("tmp.jpg")
        try:
            preds = fa.get_landmarks(input)[-1]
        except:
            preds = preds
        

        # threedleft = preds[]
        print(preds.shape)

        print("Pose Estimation")
        print(headpos)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # print(shape.shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]   
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            # perclos parameter

            if(ear < EYE_AR_THRESH):
                # frame_per_count = frame_per_count + 1
                if(frame_queue.qsize() < 30):
                    frame_queue.put(1)
                else:
                    frame_queue.get()
                    frame_queue.put(1)
            else:
                if(frame_queue.qsize() < 30):
                    frame_queue.put(0)
                else:
                    frame_queue.get()
                    frame_queue.put(0)

            leftIris = get_pupil_location(leftEye)
            rightIris = get_pupil_location(rightEye)
            # print(leftIris)
            # print(rightIris)
            left_Iris = []
            right_Iris = []
            for i, i_tmp in enumerate(leftIris):
                left_Iris.append(int(i_tmp))
            for i, i_tmp in enumerate(rightIris):
                right_Iris.append(int(i_tmp))
            # print(left_Iris)

            cv2.circle(frame, tuple(left_Iris), 2, [0, 0, 255], -1) #left_iris
            cv2.circle(frame, tuple(right_Iris), 2, [0, 255, 0], -1) #right_iris

            

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            print(leftEyeHull.shape)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


            # blink_detection
            # if ear < EYE_AR_THRESH:
            #     COUNTER += 1
            # else:
            #     if(COUNTER >= EYE_AR_CONSEC_FRAMES):
            #         TOTAL += 1
            #     COUNTER = 0
            # cv2.putText(frame, "blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # print("Gaze estimation")
        # predict_gaze = estimate_gaze(headpos, leftEye, preds, True)
        # print(predict_gaze)

        # gaze estimation over a period of time 
        if(headpos[0] > 30 or headpos[1] > 30 or headpos[2]>30 ):
                # frame_per_count = frame_per_count + 1
            if(distract.qsize() < 30):
                distract.put(False)
            else:
                distract.get()
                distract.put(False)
        else:
            if(distract.qsize() < 30):
                distract.put(True)
            else:
                distract.get()
                distract.put(True)
        distract, gazedist = distract_detect(distract)

        
        
        cv2.imshow("Frame", frame)
        frame_queue_tmp = Queue()
        # print(frame_queue.qsize())
        frame_per_count = 0
        for queue_cnt in range(0, frame_queue.qsize()):
            tmp_queue = frame_queue.get()
            frame_queue_tmp.put(tmp_queue)
            if(tmp_queue == 1):
                frame_per_count = frame_per_count + 1
        frame_queue = frame_queue_tmp
        print("the eye closure over the time: ")
        if(frame_queue.qsize() > 0):
            print(frame_per_count / (1.0 * frame_queue.qsize()))

        judge_fatigue(frame_per_count / (1.0 * frame_queue.qsize()), gazedist)
        # out.write(frame)
        # print(frame.shape)
        videoWriter.write(frame)
        key = cv2.waitKey(100) & 0xFF

        if(key == ord('q')):
            break
    cv2.destroyAllWindows()
    vs.stop()
    # out.release()
    videoWriter.release()




