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

        # if(frame_cnt % 5 == 0):
        if True:
            cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = cnn_face_detector(cv2_frame, 1)
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
                    # Print new frame with cube and axis
                    # txt_out.write(str(frame_cnt) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                    if(frame_cnt > 1):
                        utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)

        if(frame_cnt > 1):
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)


        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]   
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            print(leftEyeHull.shape)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if(COUNTER >= EYE_AR_CONSEC_FRAMES):
                    TOTAL += 1
                COUNTER = 0
            cv2.putText(frame, "blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
        cv2.imshow("Frame", frame)
        # out.write(frame)
        print(frame.shape)
        videoWriter.write(frame)
        key = cv2.waitKey(100) & 0xFF

        if(key == ord('q')):
            break
    cv2.destroyAllWindows()
    vs.stop()
    # out.release()
    videoWriter.release()




