import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
import time
import pandas as pd
import os

import torch
import torch.nn as nn

class Reshaper(nn.Module):
    def __init__(self, target_shape):
        super(Reshaper, self).__init__()
        self.target_shape = target_shape

    def forward(self, input):
        return torch.reshape(input, (-1, *self.target_shape))


class EyesNet(nn.Module):
    def __init__(self):
        super(EyesNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            Reshaper([64])
        )

        self.fc_pupil = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
        self.fc_corner1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )
        self.fc_corner2= nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x_pupil = self.fc_pupil(x)
        x_corner1 = self.fc_corner1(x)
        x_corner2 = self.fc_corner2(x)

        return x_pupil, x_corner1, x_corner2


class Eye:
    # x - 0, y - 1
    def __init__(self):
        self.center = [0, 0]
        self.inside = [0, 0]
        self.outside = [0, 0]
        self.top = [0, 0]
        self.bottom = [0, 0]
        self.out_distance_x, inner_distance_x = 0, 0
    
    def draw(self, frame):
        cv2.drawMarker(frame, (self.center[0], self.center[1]), (255, 255, 0), markerSize=5)
        cv2.drawMarker(frame, (self.inside[0], self.inside[1]), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.outside[0], self.outside[1]), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.top[0], self.top[1]), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.bottom[0], self.bottom[1]), (255, 0, 255), markerSize=5)
        return frame
    
class EyeDistances:
    def __init__(self):
        self.left_eye = Eye()
        self.right_eye = Eye()
        self.standard_x = 0
        self.out_distance_avg_x, self.inner_distance_avg_x = 0, 0
        self.distance_percentage_x = 0 #out/inner

    def get_distance(self):
        self.standard_x = self.left_eye.center[0] - self.right_eye.center[0]
        self.left_eye.out_distance_x = (self.left_eye.outside[0] - self.left_eye.center[0]) / self.standard_x
        self.right_eye.out_distance_x = (self.right_eye.center[0] - self.right_eye.outside[0]) / self.standard_x
        self.out_distance_avg_x = (self.left_eye.out_distance_x + self.right_eye.out_distance_x) / 2

        self.left_eye.inner_distance_x = (self.left_eye.center[0] - self.left_eye.inside[0]) / self.standard_x
        self.right_eye.inner_distance_x = (self.right_eye.inside[0] - self.right_eye.center[0]) / self.standard_x
        self.inner_distance_avg_x = (self.left_eye.inner_distance_x + self.right_eye.inner_distance_x) / 2

        self.distance_percentage_x = self.out_distance_avg_x / self.inner_distance_avg_x


class EyesDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                refine_landmarks=True,
                                                max_num_faces=2,
                                                min_detection_confidence=0.5)

    def get_face_mesh_results(self, frame):
        return self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_eyes_coordinates(self, results, frame, eye_distances):
        frame_h, frame_w = frame.shape
        for face_landmarks in results.multi_face_landmarks: #if several faces
            eye_distances.right_eye.center = [int(face_landmarks.landmark[468].x * frame_w), int(face_landmarks.landmark[468].y * frame_h)]
            eye_distances.left_eye.center = [int(face_landmarks.landmark[473].x * frame_w), int(face_landmarks.landmark[473].y * frame_h)]
            eye_distances.right_eye.inside = [int(face_landmarks.landmark[133].x * frame_w), int(face_landmarks.landmark[133].y * frame_h)]
            eye_distances.left_eye.inside = [int(face_landmarks.landmark[362].x * frame_w), int(face_landmarks.landmark[362].y * frame_h)]
            eye_distances.right_eye.outside = [int(face_landmarks.landmark[33].x * frame_w), int(face_landmarks.landmark[33].y * frame_h)]
            eye_distances.left_eye.outside = [int(face_landmarks.landmark[263].x * frame_w), int(face_landmarks.landmark[263].y * frame_h)]
            eye_distances.right_eye.top = [int(face_landmarks.landmark[159].x * frame_w), int(face_landmarks.landmark[159].y * frame_h)]
            eye_distances.left_eye.top = [int(face_landmarks.landmark[386].x * frame_w), int(face_landmarks.landmark[386].y * frame_h)]
            eye_distances.right_eye.bottom = [int(face_landmarks.landmark[145].x * frame_w), int(face_landmarks.landmark[145].y * frame_h)]
            eye_distances.left_eye.bottom = [int(face_landmarks.landmark[374].x * frame_w), int(face_landmarks.landmark[374].y * frame_h)]

            eye_distances.get_distance()

            return eye_distances



class CalibrationData:
    # Left corner top
    lct_left_eye, lct_right_eye = Eye(), Eye()
    tc_left_eye, tc_right_eye = Eye(), Eye()
    rct_left_eye, rct_right_eye = Eye(), Eye()


    def __init__(self):
        i = 0
        cap = cv2.VideoCapture(0)
        eyeDetector = EyesDetector()

        while True:
            ret, frame = cap.read()
            # results = eyeDetector.get_face_mesh_results(frame)
            # if not results.multi_face_landmarks:
            #     continue
            # eyeDetector.get_eyes_coordinates(results, frame)
            # frame_right_eye = eyeDetector.right_eye.draw(frame)
            # result_frame = eyeDetector.left_eye.draw(frame_right_eye)
            # result_frame = cv2.flip(result_frame, 1)
            cv2.imshow('calibrate eyes', frame)

            if cv2.waitKey(1) & 0xFF == ord('a'):
                # lct_left_eye, lct_right_eye = eyeDetector.left_eye, eyeDetector.right_eye
                cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/calibration/result' + str(i) + '.jpg', frame)
                i += 1
                # cap.release()
                # cv2.destroyWindow('calibrate eyes')
                # cv2.destroyAllWindows()
                if i == 9:
                    break

def PreDefine(eyeDetector):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, frame = cv2.split(hsv)
    cv2.imwrite('frame2.jpg', frame)
    results = eyeDetector.get_face_mesh_results(frame)
    eyeDetector.get_eyes_coordinates(results, frame)
    eye_difference = eyeDetector.left_eye.outside[0] - eyeDetector.right_eye.outside[0]
    eye_right_x = eyeDetector.right_eye.inside[0] - eyeDetector.right_eye.outside[0]
    eye_left_x = eyeDetector.left_eye.outside[0] - eyeDetector.left_eye.inside[0]
    cap.release()
    return eye_difference, eye_right_x, eye_left_x, eyeDetector.left_eye, frame

def Contrast(image):
    # image = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye3/result0.jpg', 1)
    # cv2.imshow("Original image", image)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=60., tileGridSize=(6, 6))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2, a, b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2
    # cv2.imshow('Increased contrast', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def Fixate(image0, x0, y0, image1):
    n = 7
    # image0 = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye_v_without_glasses/result0' + '.jpg',1)
    # cap = cv2.VideoCapture(0)
    # eyeDetector = EyesDetector()
    #
    # ret, image0 = cap.read()
    # ret, image0 = cap.read()
    # ret, image0 = cap.read()
    # hsv = cv2.cvtColor(image0, cv2.COLOR_BGR2HSV)
    # _, _, image0 = cv2.split(hsv)
    # cv2.imwrite('frame.jpg', image0)
    #
    # results = eyeDetector.get_face_mesh_results(image0)
    # eyeDetector.get_eyes_coordinates(results, image0)
    # x0, y0 = eyeDetector.left_eye.outside_x, eyeDetector.left_eye.outside_y
    #
    # # image1 = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye_v_without_glasses/result1' + '.jpg', 1)
    # ret, image1 = cap.read()
    # hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    # _, _, image1 = cv2.split(hsv1)
    #
    #
    # results = eyeDetector.get_face_mesh_results(image1)
    # eyeDetector.get_eyes_coordinates(results, image1)
    # x1, y1 = eyeDetector.left_eye.outside_x, eyeDetector.left_eye.outside_y
    diff = np.zeros((n, n),dtype = np.int_)

    image0 = image0.astype(np.int16)
    image1 = image1.astype(np.int16)

    halfn = n//2
    x, y = y0 - halfn, x0 - halfn
    # print('image0: ', x0, y0)
    # print('image1: ', x1, y1)


    for a in range(-halfn, halfn+1):
        for b in range(-halfn, halfn+1):
            for i in range(0, n):
                for j in range(0, n):
                    k = abs(i - halfn) + abs(j - halfn) + 1
                    diff[halfn+a, halfn+b] += abs(image0[x+i][y+j] - image1[x+i+a][y+j+b]) * k

    i = 0
    # cv2.imwrite(
    #     '/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/difference' + '.jpg',
    #     diff)
    print(np.matrix(diff))
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # X, Y = np.meshgrid(range(0, n), range(0, n))
    # ax.plot_surface(X, Y, diff)
    # plt.show()

    # cap.release()
    coordinates_of_min = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
    new_x, new_y = x0 - (halfn - coordinates_of_min[1]), y0 - (halfn - coordinates_of_min[0])
    # print(coordinates_of_min)
    print(new_x, new_y)
    # new_image = (image0 + image1) / 2

    return new_x, new_y, image1


def main():
    #calibrationData = CalibrationData()
    cap = cv2.VideoCapture(0)
    eyeDetector = EyesDetector()

    eye_difference, eye_right_delta_x, eye_left_delta_x, left_eye, image0 = PreDefine(eyeDetector)
    x0, y0 = left_eye.outside[0], left_eye.outside[1]
    # x0, y0 = 1084, 759
    # image0 = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye_fixed/result' + str(0) + '.jpg')
    # hsv = cv2.cvtColor(image0, cv2.COLOR_BGR2HSV)
    # _, _, image0 = cv2.split(hsv)
    # cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye_fixed/result' + str(0) + '.jpg', image0)
    print('x0, y0', x0, y0)
    # print(eye_difference)
    i = 1

    while True:
        ret, image1 = cap.read()

        # image1 = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye_fixed/result' + str(i) + '.jpg')
        hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        _, _, image1 = cv2.split(hsv)
        # start_time = time.time()
        x0, y0, image0 = Fixate(image0, x0, y0, image1)
        # cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye_fixed_b/new_image' + str(i) + '.jpg',
        #             image0[eyeDetector.left_eye.outside_y - 40:eyeDetector.left_eye.outside_y + 20,
        #                eyeDetector.left_eye.outside_x - 20 -40: eyeDetector.left_eye.outside_x + 15])
        # end_time = time.time()
        # execute_time = end_time - start_time
        # print(execute_time)
        # image0 = image1.copy()
        # cv2.drawMarker(image1, (x0, y0), (255, 0, 255), markerSize=5)

        # To improve performance
        # frame.flags.writeable = False

        # frame = Contrast(frame)
        # results = eyeDetector.get_face_mesh_results(image1)
        cv2.drawMarker(image1, (x0, y0), (255, 0, 255), markerSize=5)
        # Make it writeable again
        # frame.flags.writeable = True

        # if not results.multi_face_landmarks:
        #     continue
        #
        # eyeDetector.get_eyes_coordinates(results, image1)
        # frame_right_eye = eyeDetector.right_eye.draw(frame)
        # result_frame = eyeDetector.left_eye.draw(frame_right_eye)

        # result_frame = cv2.flip(result_frame, 1)
        # Show the image
        # cv2.imshow('image1', image1)
        # cv2.waitKey()
        # result_frame = result_frame[eyeDetector.right_eye.outside_y-30:eyeDetector.right_eye.outside_y+30,
        #                eyeDetector.right_eye.outside_x-10:eyeDetector.right_eye.outside_x+10+eye_difference]
        result_frame = image1[y0 - 40:y0 + 20,
                       x0 - 20 - 40:x0 + 15]
        # result_frame = image1[eyeDetector.left_eye.outside_y - 40:eyeDetector.left_eye.outside_y + 20,
        #                eyeDetector.left_eye.outside_x - 20 -40: eyeDetector.left_eye.outside_x + 15]
        # print('image0: ', x0, y0)
        cv2.imshow('eyes difference', result_frame)
        # cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye_fixed_b/result' + str(i) + '.jpg', result_frame)
        # i += 1
        # if i == 3:
        #     break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def Test():
    # image0 = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/test/bbb' + '.jpeg',
    #                     1)
    # cap = cv2.VideoCapture(0)
    # ret, image1 = cap.read()
    # hsv = cv2.cvtColor(image1, cv2)
    # _, image1 = cv2.split(hsv)
    # cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/frame_for_testing/result0' + '.jpg')
    image0 = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/video_as_photos/result0' + '.jpg',
                        1)
    hsv = cv2.cvtColor(image0, cv2.COLOR_BGR2HSV)
    _, _, image0 = cv2.split(hsv)
    eyeDetector = EyesDetector()
    results = eyeDetector.get_face_mesh_results(image0)
    eyeDetector.get_eyes_coordinates(results, image0)
    x0, y0 = eyeDetector.left_eye.outside_x, eyeDetector.left_eye.outside_y
    # cv2.drawMarker(image0, (x0, y0), (255, 0, 255), markerSize=5)
    # cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye3/result' + '.jpg',
    #             image0)
    for i in range(1, 20):
        image1 = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/video_as_photos/result' + str(i) + '.jpg',
                            1)

        hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        _, _, image1 = cv2.split(hsv)
        image2 = image0.copy()
        # print(image0)
        #
        x0, y0, image0 = Fixate(image0, x0, y0, image1)
        cv2.drawMarker(image1, (x0, y0), (255, 0, 255), markerSize=5)
        # cv2.imshow('image1', image1)
        cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye5/result' + str(i) + '.jpg',
                    image1[y0-15:y0+15, x0 -50 : x0+ 20])
        # x0, y0, image0 = Fixate(image0, x0, y0, image2[1:60, :])
        # cv2.drawMarker(image0, (x0, y0), (255, 0, 255), markerSize=5)
        # cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/test_of_cut_frame/result' + str(i) + '.jpeg',
        #             image0[y0 - 15:y0 + 15, x0 - 20:x0 + 20])
    # image1 = np.array([[0,  1,   7,   0,   0,   3,   2, 0, 0, 248, 255, 255, 254, 255, 248,   7,   0,   1,   0,   0,   0],
    #           [0,  1,   7,   0,   0,   3,   2, 0, 0, 248, 255, 255, 254, 255, 248,   7,   0,   1,   0,   0,   0],
    #           [0,  1,   7,   0,   0,   3,   2, 0, 0, 248, 255, 255, 254, 255, 248,   7,   0,   1,   0,   0,   0],
    #           [0,  1,   7,   0,   0,   3,   2, 0, 0, 248, 255, 255, 254, 255, 248,   7,   0,   1,   0,   0,   0],
    #           [0,  1,   7,   0,   0,   3,   2, 0, 0, 248, 255, 255, 254, 255, 248,   7,   0,   1,   0,   0,   0],
    #           [0,  1,   7,   0,   0,   3,   2, 0,  0,248, 255, 255, 254, 255, 248,   7,   0,   1,   0,   0,   0],
    #           [0,  1,   7,   0,   0,   3,   2, 0, 0, 248, 255, 255, 254, 255, 248,   7,   0,   1,   0,   0,   0],
    #           [0, 1, 7, 0, 0, 3, 2, 0, 0, 248, 255, 255, 254, 255, 248, 7, 0, 1, 0, 0, 0],
    #           [0, 1, 7, 0, 0, 3, 2, 0, 0, 248, 255, 255, 254, 255, 248, 7, 0, 1, 0, 0, 0]])
    # cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/test/bbb' + '.jpeg', image1)

def TestWithLiveVideo():
    cap = cv2.VideoCapture(0)
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    ret, image0 = cap.read()
    hsv = cv2.cvtColor(image0, cv2.COLOR_BGR2HSV)
    _, _, image0 = cv2.split(hsv)
    eyeDetector = EyesDetector()
    results = eyeDetector.get_face_mesh_results(image0)
    eyeDetector.get_eyes_coordinates(results, image0)
    x0, y0 = eyeDetector.left_eye.outside_x, eyeDetector.left_eye.outside_y

    for i in range(1, 20):
        ret, image1 = cap.read()
        hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        _, _, image1 = cv2.split(hsv)
        x0, y0, _= Fixate(image0, x0, y0, image1)
        image0 = image1.copy()
        # image2 = image0.copy()
        cv2.drawMarker(image1, (x0, y0), (255, 0, 255), markerSize=5)
        cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/live_video_results/result' + str(i) + '.jpg',image1)

def GetCornerPixels():
    image0 = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/test_photos_window_on_side/new_image1.jpg')
    hsv = cv2.cvtColor(image0, cv2.COLOR_BGR2HSV)
    _, _, image0 = cv2.split(hsv)

    x0, y0 = 21, 21

    image1 = image0[y0-5:y0+5, x0-5:x0+5]
    print(image1)
    cv2.imshow('f', image1)
    cv2.waitKey()


def GetDistance():
    image0 = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/calibration/result0.jpg', 0)
    eyeDetector = EyesDetector()
    results = eyeDetector.get_face_mesh_results(image0)
    eyeDetector.get_eyes_coordinates(results, image0)
    image1 = eyeDetector.eye_distances.right_eye.draw(image0)
    image1 = eyeDetector.eye_distances.left_eye.draw(image1)
    # standard_x = eyeDetector.left_eye.center_x - eyeDetector.right_eye.center_x
    # out_left_x = (eyeDetector.left_eye.outside_x - eyeDetector.left_eye.center_x)/standard_x
    # out_right_x = (eyeDetector.right_eye.center_x - eyeDetector.right_eye.outside_x)/standard_x
    # out_avg_x = (out_left_x + out_right_x)/2
    #
    # inner_left_x = (eyeDetector.left_eye.center_x - eyeDetector.left_eye.inside_x)/standard_x
    # inner_right_x = (eyeDetector.right_eye.inside_x - eyeDetector.right_eye.center_x)/standard_x
    # inner_avg_x = (inner_left_x + inner_right_x)/2

    print(eyeDetector.eye_distances.standard_x)
    print('out:  ', eyeDetector.eye_distances.left_eye.out_distance_x, eyeDetector.eye_distances.right_eye.out_distance_x, eyeDetector.eye_distances.out_distance_avg_x)
    print('inner:', eyeDetector.eye_distances.left_eye.inner_distance_x, eyeDetector.eye_distances.right_eye.inner_distance_x, eyeDetector.eye_distances.inner_distance_avg_x)

    print('percentage: ', eyeDetector.eye_distances.distance_percentage_x)

    print('left eye:  ', eyeDetector.eye_distances.left_eye.out_distance_x, eyeDetector.eye_distances.left_eye.inner_distance_x)
    print('right eye: ', eyeDetector.eye_distances.right_eye.out_distance_x, eyeDetector.eye_distances.right_eye.inner_distance_x)

    # cv2.imshow('aa', image1)
    # cv2.imshow('flipped', cv2.flip(image1, 1))
    # cv2.waitKey()

def Distance(x1, y1, x2, y2):
    return int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

image_shape = (16, 32)

def GetEyesImage(image, corner1, corner2, pupil):
    line_len = Distance(*corner1, *corner2)
    corner_of_frame1 = corner1 - np.array([line_len // 4, line_len // 8]) #- np.array([5, 10])
    corner_of_frame2 = corner2 + np.array([line_len // 4, line_len // 8]) #+ np.array([5, 10])

    sub_image = image[corner_of_frame1[0]:corner_of_frame2[0], corner_of_frame1[1]:corner_of_frame2[1]]
    sub_shape = sub_image.shape

    pupil_new = pupil - corner_of_frame1
    corner1_new = corner1 - corner_of_frame1
    corner2_new = corner2 - corner_of_frame1
    pupil_new = pupil_new / sub_image.shape[:2]
    corner1_new = corner1_new / sub_image.shape[:2]
    corner2_new = corner2_new / sub_image.shape[:2]

    sub_image = cv2.resize(sub_image, image_shape[::-1], interpolation=cv2.INTER_AREA)
    sub_shape = sub_shape[0] / sub_image.shape[0], sub_shape[1] / sub_image.shape[1]
    sub_image = np.array(sub_image) / 255.0
    sub_image = torch.FloatTensor(sub_image).unsqueeze(0)
    # , pupil_new, corner1_new, corner2_new
    return sub_image, corner_of_frame1, sub_shape

def ResizeCoordinates(coordinate, shapes_diff, corner_of_frame):
    coordinate = coordinate[0] * image_shape
    coordinate *= shapes_diff
    coordinate += corner_of_frame

    coordinate = round(coordinate[1]), round(coordinate[0])
    return coordinate

def GetEyesCoordinates(eyesnet_left, eyesnet_right, frame, eye_distances):
    image_left, corner_of_frame_left, shapes_diff = GetEyesImage(frame, eye_distances.left_eye.inside[::-1],
                                                                 eye_distances.left_eye.outside[::-1],
                                                                 eye_distances.left_eye.center[::-1])
    y_pupil_left, y_corner1_left, y_corner2_left = eyesnet_left(image_left)

    y_pupil_left = y_pupil_left.detach().squeeze().numpy().reshape(-1, 2)
    y_corner1_left = y_corner1_left.detach().squeeze().numpy().reshape(-1, 2)
    y_corner2_left = y_corner2_left.detach().squeeze().numpy().reshape(-1, 2)

    image_left = np.hstack(image_left)
    plt.imshow(image_left)
    plt.scatter(y_pupil_left[0, 1] * image_shape[1], y_pupil_left[0, 0] * image_shape[0], c="r")
    plt.scatter(y_corner1_left[0, 1] * image_shape[1], y_corner1_left[0, 0] * image_shape[0], c="r")
    plt.scatter(y_corner2_left[0, 1] * image_shape[1], y_corner2_left[0, 0] * image_shape[0], c="r")
    plt.show()


    eye_distances.left_eye.center = (np.array(ResizeCoordinates(y_pupil_left, shapes_diff, corner_of_frame_left)) + np.array(eye_distances.left_eye.center)) // 2
    eye_distances.left_eye.inside = (np.array(ResizeCoordinates(y_corner1_left, shapes_diff, corner_of_frame_left)) + np.array(eye_distances.left_eye.inside)) // 2
    eye_distances.left_eye.outside = (np.array(ResizeCoordinates(y_corner2_left, shapes_diff, corner_of_frame_left)) + np.array(eye_distances.left_eye.outside)) // 2

    image_right, corner_of_frame_right, shapes_diff = GetEyesImage(frame, eye_distances.right_eye.outside[::-1],
                                                                   eye_distances.right_eye.inside[::-1],
                                                                   eye_distances.right_eye.center[::-1])
    y_pupil_right, y_corner1_right, y_corner2_right = eyesnet_right(image_right)

    y_pupil_right = y_pupil_right.detach().squeeze().numpy().reshape(-1, 2)
    y_corner1_right = y_corner1_right.detach().squeeze().numpy().reshape(-1, 2)
    y_corner2_right = y_corner2_right.detach().squeeze().numpy().reshape(-1, 2)

    eye_distances.right_eye.center = (np.array(ResizeCoordinates(y_pupil_right, shapes_diff, corner_of_frame_right)) + np.array(eye_distances.right_eye.center)) // 2
    eye_distances.right_eye.outside = (np.array(ResizeCoordinates(y_corner1_right, shapes_diff, corner_of_frame_right)) + np.array(eye_distances.right_eye.outside)) // 2
    eye_distances.right_eye.inside = (np.array(ResizeCoordinates(y_corner2_right, shapes_diff, corner_of_frame_right)) + np.array(eye_distances.right_eye.inside)) // 2

    return eye_distances

def NetTry():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    ret, frame = cap.read()
    eyesnet = EyesNet()
    eyesnet.load_state_dict(torch.load("eyes_net/epoch_300.pth"))

    eye_distances = EyeDistances()
    eyeDetector = EyesDetector()
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, frame = cv2.split(hsv)

    frame.flags.writeable = False
    results = eyeDetector.get_face_mesh_results(frame)
    frame.flags.writeable = True
    eye_distances = eyeDetector.get_eyes_coordinates(results, frame, eye_distances)

    eye_distances = GetEyesCoordinates(eyesnet, frame, eye_distances)
    # cv2.drawMarker(frame, eye_distances.left_eye.center, (255, 0, 255), markerSize=5)
    # cv2.drawMarker(frame, eye_distances.left_eye.inside, (255, 0, 255), markerSize=5)
    # cv2.drawMarker(frame, eye_distances.left_eye.outside, (255, 0, 255), markerSize=5)
    #
    # cv2.drawMarker(frame, eye_distances.right_eye.center, (255, 0, 255), markerSize=5)
    # cv2.drawMarker(frame, eye_distances.right_eye.inside, (255, 0, 255), markerSize=5)
    # cv2.drawMarker(frame, eye_distances.right_eye.outside, (255, 0, 255), markerSize=5)

    # eyeDetector.eye_distances.left_eye.draw(frame)
    # cv2.imshow("Pupil", frame)
    # cv2.waitKey()


    for i in range(20):
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, _, frame = cv2.split(hsv)
        eye_distances = GetEyesCoordinates(eyesnet, frame, eye_distances)
        # results = eyeDetector.get_face_mesh_results(frame)
        # eye_distances = eyeDetector.get_eyes_coordinates(results, frame, eye_distances)
        # cv2.drawMarker(frame, eye_distances.left_eye.center, (255, 0, 255), markerSize=5)
        # cv2.drawMarker(frame, eye_distances.left_eye.inside, (255, 0, 255), markerSize=5)
        # cv2.drawMarker(frame, eye_distances.left_eye.outside, (255, 0, 255), markerSize=5)
        #
        # cv2.drawMarker(frame, eye_distances.right_eye.center, (255, 0, 255), markerSize=5)
        # cv2.drawMarker(frame, eye_distances.right_eye.inside, (255, 0, 255), markerSize=5)
        # cv2.drawMarker(frame, eye_distances.right_eye.outside, (255, 0, 255), markerSize=5)
        cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/mynetTry1/result' + str(i)+'.jpg', frame)

    cap.release()

def DebugNet():
    eyesnet_left = EyesNet()
    eyesnet_left.load_state_dict(torch.load("/Users/illaria/BSUIR/Diploma/code/PyTorchTry1/eyes_net_left_my_dataset_fixed_photos/epoch_400.pth"))
    eyesnet_right = EyesNet()
    eyesnet_right.load_state_dict(torch.load("/Users/illaria/BSUIR/Diploma/code/PyTorchTry1/eyes_net_right_my_dataset_fixed_photos/epoch_400.pth"))

    eye_distances = EyeDistances()
    eyeDetector = EyesDetector()
    frame = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/photos_moving_pupils_much_left_first/result0.jpg', 0)
    results = eyeDetector.get_face_mesh_results(frame)
    eye_distances = eyeDetector.get_eyes_coordinates(results, frame, eye_distances)
    plt.imshow(frame)
    plt.scatter(eye_distances.left_eye.center[0], eye_distances.left_eye.center[1], c="r")
    plt.scatter(eye_distances.left_eye.inside[0], eye_distances.left_eye.inside[1], c="r")
    plt.scatter(eye_distances.left_eye.outside[0], eye_distances.left_eye.outside[1], c="r")
    plt.show()
    for i in range(1, 20):
        frame = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/photos_moving_pupils_much_left_first/result' + str(i)+'.jpg', 0)
        if i % 5 == 0:
            results = eyeDetector.get_face_mesh_results(frame)
            # prev_eye_distances = eye_distances
            eye_distances = eyeDetector.get_eyes_coordinates(results, frame, eye_distances)
            # eye_distances.right_eye.outside = (prev_eye_distances.right_eye.outside + eye_distances.right_eye.outside)/2
        eye_distances = GetEyesCoordinates(eyesnet_left, eyesnet_right, frame, eye_distances)
        cv2.drawMarker(frame, eye_distances.left_eye.center, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.left_eye.inside, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.left_eye.outside, (255, 0, 255), markerSize=5)

        cv2.drawMarker(frame, eye_distances.right_eye.center, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.right_eye.inside, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.right_eye.outside, (255, 0, 255), markerSize=5)
        cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/mynet_myds_fixed_moving_much_left_first_plus_google/result' + str(i) + '.jpg', frame)

# main()
# CalibrationData()
# Contrast()
# Fixate()
# Test()
# TestWithLiveVideo()
# GetCornerPixels()
# GetDistance()
# NetTry()
DebugNet()


# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # ret, frame = cv2.threshold(frame, 127, 255, 0)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     _, _, frame = cv2.split(hsv)
#     cv2.imshow('eyes', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyWindow()

def handle_eye(image, corner1, corner2, pupil):

    # line_len = distance(*corner1, *corner2)
    # x, y -> y, x
    corner1 = corner1[::-1]
    corner2 = corner2[::-1]
    pupil = pupil[::-1]

    # if corner1[0] > corner2[0]:
    #     corner1[0], corner2[0] = corner2[0], corner1[0]

    corner_of_frame1 = corner1 - np.array([40, 20]) #- np.array([5, 10])
    corner_of_frame2 = corner2 + np.array([30, 20]) #+ np.array([5, 10])

    sub_image = image[corner_of_frame1[0]:corner_of_frame2[0], corner_of_frame1[1]:corner_of_frame2[1]]

    pupil_new = pupil - corner_of_frame1
    corner1_new = corner1 - corner_of_frame1
    corner2_new = corner2 - corner_of_frame1
    pupil_new = pupil_new / sub_image.shape[:2]
    corner1_new = corner1_new / sub_image.shape[:2]
    corner2_new = corner2_new / sub_image.shape[:2]

    # sub_image = cv2.resize(sub_image, image_shape[::-1], interpolation=cv2.INTER_AREA)
    # sub_image = cv2.cvtColor(sub_image, cv2.COLOR_RGB2GRAY)

    return sub_image, pupil_new, corner1_new, corner2_new

def my_image_to_train_data(image, points):
    eye_right_c1 = points[6:8]
    eye_right_c2 = points[8:10]
    eye_right_pupil = points[10:12]
    eye_right_c1 = eye_right_c1[::-1]
    eye_right_c2 = eye_right_c2[::-1]
    eye_right_pupil = eye_right_pupil[::-1]

    right_image, right_pupil, right_corner1, right_corner2 = handle_eye(image, eye_right_c1, eye_right_c2,
                                                                        eye_right_pupil)

    eye_left_c1 = points[0:2]
    eye_left_c2 = points[2:4]
    eye_left_pupil = points[4:6]
    eye_left_c1 = eye_left_c1[::-1]
    eye_left_c2 = eye_left_c2[::-1]
    eye_left_pupil = eye_left_pupil[::-1]

    left_image, left_pupil, left_corner1, left_corner2 = handle_eye(image, eye_left_c1, eye_left_c2, eye_left_pupil)

    return right_image, right_pupil, right_corner1, right_corner2, left_image, left_pupil, left_corner1, left_corner2

def load_my_image_data(patient_name):

    annotation_path = os.path.join('/Users/illaria/BSUIR/Diploma/mydataset_only_good_photos/', patient_name + '/annotationscopy.txt')
    data_folder = os.path.join('/Users/illaria/BSUIR/Diploma/mydataset_only_good_photos/', patient_name)

    annotation = pd.read_csv(annotation_path, sep=" ", header=None)

    points = np.array(annotation.loc[:, list(range(0, 12))])
    print(points)

    filenames = os.listdir(data_folder)
    filenames.sort()
    filenames.remove('annotations.txt')
    filenames.remove('annotationscopy.txt')
    filenames.remove('.DS_Store')
    images = [np.array(Image.open(os.path.join(data_folder, filename))) for filename in filenames]

    return images, points

def get_my_image_data(images, points):
    for i in range(len(images)):
        signle_image_data = my_image_to_train_data(images[i], points[i])

        if any(stuff is None for stuff in signle_image_data):
            continue

        right_image, right_pupil, right_corner1, right_corner2, left_image, left_pupil, left_corner1, left_corner2 = signle_image_data
        if i == 68:
            # cv2.drawMarker(images[i], [points[i][1], points[i][0]], (255, 0, 255), markerSize=5)
            # cv2.drawMarker(images[i], [points[i][3], points[i][2]], (255, 0, 255), markerSize=5)
            # cv2.drawMarker(images[i], [points[i][5], points[i][4]], (255, 0, 255), markerSize=5)
            plt.imshow(left_image)
            plt.title('left_pupil' + str(i))
            plt.scatter(left_pupil[1] * left_image.shape[1], left_pupil[0] * left_image.shape[0], c="r")  # [0] - y, [1] - x
            plt.scatter(left_corner1[1] * left_image.shape[1], left_corner1[0] * left_image.shape[0], c="r")
            plt.scatter(left_corner2[1] * left_image.shape[1], left_corner2[0] * left_image.shape[0], c="r")
            plt.show()
            plt.imshow(right_image)
            plt.title('right_pupil' + str(i))
            plt.scatter(right_pupil[1] * right_image.shape[1], right_pupil[0] * right_image.shape[0], c="r")  # [0] - y, [1] - x
            plt.scatter(right_corner1[1] * right_image.shape[1], right_corner1[0] * right_image.shape[0], c="r")
            plt.scatter(right_corner2[1] * right_image.shape[1], right_corner2[0] * right_image.shape[0], c="r")
            plt.show()

        if any(right_pupil < 0) or any(left_pupil < 0):
            continue

# images, points = load_my_image_data('15')
# get_my_image_data(images, points)
# cv2.drawMarker(images[0], [points[0][1] - 5, points[0][0] - 4], (255, 0, 255), markerSize=5)
# cv2.drawMarker(images[0], [points[0][3], points[0][2]], (255, 0, 255), markerSize=5)
# cv2.drawMarker(images[0], [points[0][5], points[0][4]], (255, 0, 255), markerSize=5)
# cv2.drawMarker(images[0], [points[0][7] - 2, points[0][6] - 1], (255, 0, 255), markerSize=5)
# cv2.drawMarker(images[0], [points[0][9] - 3, points[0][8]-3] , (255, 0, 255), markerSize=5)
# cv2.drawMarker(images[0], [points[0][11], points[0][10]], (255, 0, 255), markerSize=5)
# cv2.imshow('aa', images[0])
# cv2.waitKey(0)

# annotation = pd.read_csv('/Users/illaria/BSUIR/Diploma/mydataset/01/annotations.txt', sep=" ", header=None)
# points = np.array(annotation.loc[:, list(range(2))])
# filenames = np.array(annotation.loc[:, [0]]).reshape(-1)
# images = [np.array(Image.open(os.path.join(data_folder, filename))) for filename in filenames]
# print(points)

def CreateMyDataset():
    cap = cv2.VideoCapture(0)
    eye_distances = EyeDistances()
    eyeDetector = EyesDetector()

    ret, frame = cap.read()
    ret, frame = cap.read()
    annotations = open('/Users/illaria/BSUIR/Diploma/mydataset/23/annotations.txt', 'x')
    j = 0
    for i in range(200):
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, _, frame = cv2.split(hsv)
        cv2.imshow('eyes', frame)
        if i % 2 == 0:
            results = eyeDetector.get_face_mesh_results(frame)
            eye_distances = eyeDetector.get_eyes_coordinates(results, frame, eye_distances)
            print(eye_distances.left_eye.bottom[1] - eye_distances.left_eye.top[1])
            if eye_distances.left_eye.bottom[1] - eye_distances.left_eye.top[1] > 14:
                cv2.imwrite('/Users/illaria/BSUIR/Diploma/mydataset/23/' + str(j) + '.jpg', frame)
                annotations.write(str(eye_distances.left_eye.inside[1]) + ' ' + str(eye_distances.left_eye.inside[0]) + ' ' +
                                  str(eye_distances.left_eye.outside[1]) + ' ' + str(eye_distances.left_eye.outside[0]) + ' ' +
                                  str(eye_distances.left_eye.center[1]) + ' ' + str(eye_distances.left_eye.center[0]) + ' ' +
                                  str(eye_distances.right_eye.outside[1]) + ' ' + str(eye_distances.right_eye.outside[0]) + ' ' +
                                  str(eye_distances.right_eye.inside[1]) + ' ' + str(eye_distances.right_eye.inside[0]) + ' ' +
                                  str(eye_distances.right_eye.center[1]) + ' ' + str(eye_distances.right_eye.center[0]) + '\n')
                j += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    annotations.close()

# CreateMyDataset()
