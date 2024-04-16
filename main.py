import math

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
        self.fc_corner2 = nn.Sequential(
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


class Camera:
    def __init__(self, camera_number, address_for_write, address_for_read):
        self.camera_number = camera_number
        # add checks
        self.capture = None
        self.address_for_write = address_for_write
        self.address_for_read = address_for_read

    def read_frame(self):
        if self.capture is None:
            return None
        ret, frame = self.capture.read()
        if ret is True:
            return frame

    def start_capture(self):
        self.capture = cv2.VideoCapture(self.camera_number)

    def stop_capture(self):
        self.capture.release()

    def write_frame(self, frame):
        cv2.imwrite(self.address_for_write, frame)

    def read_frame_from_file(self, flags):
        return cv2.imread(self.address_for_read, flags)


class ImageEdit():
    default_marker_size = 5
    default_marker_color = (255, 0, 255)
    default_text_font = cv2.FONT_HERSHEY_SIMPLEX
    default_text_color = (255, 0, 0)
    default_text_thickness = 2
    default_text_font_scale = 1.0

    @staticmethod
    def draw_on_image(image, position, color, marker_size=default_marker_size):
        cv2.drawMarker(image, position, color, marker_size)

    @staticmethod
    def put_text_on_image(image, position, text, font=default_text_font, font_scale=default_text_font_scale,
                          color=default_text_color, thickness=default_text_thickness):
        cv2.putText(image, text, position, font, font_scale, color, thickness)

    @staticmethod
    def split_image(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, _, frame = cv2.split(hsv)
        return frame

    @staticmethod
    def image_to_RGB(image):
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def resize_image(image, image_shape):
        cv2.resize(image, image_shape, interpolation=cv2.INTER_AREA)





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

class Eye:
    # x - 0, y - 1
    def __init__(self):
        self.pupil = [0, 0]
        self.left_corner = [0, 0]
        self.right_corner = [0, 0]
        self.top = [0, 0]
        self.bottom = [0, 0]
        self.left_distance, self.right_distance = [0, 0], [0, 0]  # x - 0, y - 1
        self.rclc = 0.0
        self.prc, self.plc = 0.0, 0.0
        self.right_angle, self.left_angle = 0.0, 0.0

    def draw(self, frame):
        cv2.drawMarker(frame, (self.pupil[0], self.pupil[1]), (255, 255, 0), markerSize=5)
        cv2.drawMarker(frame, (self.left_corner[0], self.left_corner[1]), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.right_corner[0], self.right_corner[1]), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.top[0], self.top[1]), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.bottom[0], self.bottom[1]), (255, 0, 255), markerSize=5)
        return frame


class EyeDistances:
    def __init__(self):
        self.left_eye = Eye()
        self.right_eye = Eye()
        self.standard_x = 0
        self.left_distance_avg_x, self.right_distance_avg_x = 0, 0
        self.distance_percentage_x = 0  # left/right
        self.pupil_angle = 0.0
        self.right_corner_angle = 0.0
        self.left_corner_angle = 0.0

    @staticmethod
    def get_angle(dot1, dot2):
        return math.atan((dot1[1] - dot2[1]) / (dot1[0] - dot2[0])) * 180 / math.pi

    def get_distances(self, eye):
        eye.left_distance = [(eye.left_corner[0] - eye.pupil[0]) / self.standard_x,
                             (eye.left_corner[1] - eye.pupil[1]) / self.standard_x]
        eye.right_distance = [(eye.pupil[0] - eye.right_corner[0]) / self.standard_x,
                              (eye.pupil[1] - eye.right_corner[1] / self.standard_x)]
        rclc_angle = self.get_angle(eye.left_corner, eye.right_corner)
        eye.left_angle = self.get_angle(eye.left_corner, eye.pupil) - rclc_angle
        eye.right_angle = rclc_angle - self.get_angle(eye.right_corner, eye.pupil)
        big_angle = self.get_angle(eye.right_corner, eye.pupil) - self.get_angle(eye.left_corner, eye.pupil)
        print(big_angle)

    def get_distance(self):
        self.standard_x = self.left_eye.pupil[0] - self.right_eye.pupil[0]

        self.pupil_angle = self.get_angle(self.left_eye.pupil, self.right_eye.pupil)
        self.right_corner_angle = self.get_angle(self.left_eye.right_corner, self.right_eye.right_corner)
        self.left_corner_angle = self.get_angle(self.left_eye.left_corner, self.right_eye.left_corner)

        self.get_distances(self.left_eye)
        self.get_distances(self.right_eye)

        self.left_distance_avg_x = (self.left_eye.left_distance[0] + self.right_eye.left_distance[0]) / 2
        self.right_distance_avg_x = (self.left_eye.right_distance[0] + self.right_eye.right_distance[0]) / 2

        self.distance_percentage_x = self.left_distance_avg_x / self.right_distance_avg_x


class EyesDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                         refine_landmarks=True,
                                                         max_num_faces=2,
                                                         min_detection_confidence=0.5)

    def get_face_mesh_results(self, frame):
        return self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def get_eyes_coordinates(results, frame, eye_distances):
        frame_h, frame_w = frame.shape
        for face_landmarks in results.multi_face_landmarks:  # if several faces
            eye_distances.right_eye.pupil = [int(face_landmarks.landmark[468].x * frame_w),
                                             int(face_landmarks.landmark[468].y * frame_h)]
            eye_distances.left_eye.pupil = [int(face_landmarks.landmark[473].x * frame_w),
                                            int(face_landmarks.landmark[473].y * frame_h)]
            eye_distances.right_eye.left_corner = [int(face_landmarks.landmark[133].x * frame_w),
                                                   int(face_landmarks.landmark[133].y * frame_h)]
            eye_distances.left_eye.right_corner = [int(face_landmarks.landmark[362].x * frame_w),
                                                   int(face_landmarks.landmark[362].y * frame_h)]
            eye_distances.right_eye.right_corner = [int(face_landmarks.landmark[33].x * frame_w),
                                                    int(face_landmarks.landmark[33].y * frame_h)]
            eye_distances.left_eye.left_corner = [int(face_landmarks.landmark[263].x * frame_w),
                                                  int(face_landmarks.landmark[263].y * frame_h)]
            eye_distances.right_eye.top = [int(face_landmarks.landmark[159].x * frame_w),
                                           int(face_landmarks.landmark[159].y * frame_h)]
            eye_distances.left_eye.top = [int(face_landmarks.landmark[386].x * frame_w),
                                          int(face_landmarks.landmark[386].y * frame_h)]
            eye_distances.right_eye.bottom = [int(face_landmarks.landmark[145].x * frame_w),
                                              int(face_landmarks.landmark[145].y * frame_h)]
            eye_distances.left_eye.bottom = [int(face_landmarks.landmark[374].x * frame_w),
                                             int(face_landmarks.landmark[374].y * frame_h)]

            eye_distances.get_distance()


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
                cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/calibration/result' + str(i) + '.jpg',
                            frame)
                i += 1
                # cap.release()
                # cv2.destroyWindow('calibrate eyes')
                # cv2.destroyAllWindows()
                if i == 9:
                    break


def predefine(eyeDetector):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, frame = cv2.split(hsv)
    cv2.imwrite('frame2.jpg', frame)
    results = eyeDetector.get_face_mesh_results(frame)
    eyeDetector.get_eyes_coordinates(results, frame)
    eye_difference = eyeDetector.left_eye.right_corner[0] - eyeDetector.right_eye.right_corner[0]
    eye_right_x = eyeDetector.right_eye.left_corner[0] - eyeDetector.right_eye.right_corner[0]
    eye_left_x = eyeDetector.left_eye.right_corner[0] - eyeDetector.left_eye.left_corner[0]
    cap.release()
    return eye_difference, eye_right_x, eye_left_x, eyeDetector.left_eye, frame


def contrast(image):
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


def fixate(image0, x0, y0, image1):
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
    diff = np.zeros((n, n), dtype=np.int_)

    image0 = image0.astype(np.int16)
    image1 = image1.astype(np.int16)

    halfn = n // 2
    x, y = y0 - halfn, x0 - halfn
    # print('image0: ', x0, y0)
    # print('image1: ', x1, y1)

    for a in range(-halfn, halfn + 1):
        for b in range(-halfn, halfn + 1):
            for i in range(0, n):
                for j in range(0, n):
                    k = abs(i - halfn) + abs(j - halfn) + 1
                    diff[halfn + a, halfn + b] += abs(image0[x + i][y + j] - image1[x + i + a][y + j + b]) * k

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
    pass


def test_with_live_video():
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
        x0, y0, _ = fixate(image0, x0, y0, image1)
        image0 = image1.copy()
        # image2 = image0.copy()
        cv2.drawMarker(image1, (x0, y0), (255, 0, 255), markerSize=5)
        cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/live_video_results/result' + str(i) + '.jpg',
                    image1)


def print_distance(eye_distances):
    print(eye_distances.standard_x)
    print('percentage: ', eye_distances.distance_percentage_x)

    print('left eye:  ', eye_distances.left_eye.left_distance_x,
          eye_distances.left_eye.right_distance_x)
    print('right eye: ', eye_distances.right_eye.left_distance_x,
          eye_distances.right_eye.right_distance_x)


def calculate_distance_for_eyesnet(x1, y1, x2, y2):
    return int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)


class EyesRecognizer:
    def __init__(self, path_left, path_right):
        self.eyesnet_left = EyesNet()
        self.eyesnet_left.load_state_dict(torch.load(path_left))
        self.eyesnet_right = EyesNet()
        self.eyesnet_right.load_state_dict(torch.load(path_right))
        self.image_shape = (16, 32)

    def get_eye_image(self, image, corner1, corner2):
        line_len = calculate_distance_for_eyesnet(*corner1, *corner2)
        corner_of_frame1 = corner1 - np.array([line_len // 4, line_len // 8])  # - np.array([5, 10])
        corner_of_frame2 = corner2 + np.array([line_len // 4, line_len // 8])  # + np.array([5, 10])

        sub_image = image[corner_of_frame1[0]:corner_of_frame2[0], corner_of_frame1[1]:corner_of_frame2[1]]
        sub_shape = sub_image.shape

        sub_image = cv2.resize(sub_image, self.image_shape[::-1], interpolation=cv2.INTER_AREA)
        shapes_diff = sub_shape[0] / sub_image.shape[0], sub_shape[1] / sub_image.shape[1]
        sub_image = np.array(sub_image) / 255.0
        sub_image = torch.FloatTensor(sub_image).unsqueeze(0)
        return sub_image, corner_of_frame1, shapes_diff

    def resize_coordinates(self, coordinate, shapes_diff, corner_of_frame):
        coordinate = coordinate[0] * self.image_shape
        coordinate *= shapes_diff
        coordinate += corner_of_frame

        coordinate = round(coordinate[1]), round(coordinate[0])
        return coordinate

    def get_coordinates_for_one_eye(self, image, eye, eyesnet):
        image, corner_of_frame, shapes_diff = self.get_eye_image(image, eye.right_corner[::-1], eye.left_corner[::-1])
        y_pupil, y_right_corner, y_left_corner = eyesnet(image)

        y_pupil = y_pupil.detach().squeeze().numpy().reshape(-1, 2)
        y_right_corner = y_right_corner.detach().squeeze().numpy().reshape(-1, 2)
        y_left_corner = y_left_corner.detach().squeeze().numpy().reshape(-1, 2)

        image = np.hstack(image)
        # plt.imshow(image)
        # plt.scatter(y_pupil[0, 1] * self.image_shape[1], y_pupil[0, 0] * self.image_shape[0], c="r")
        # plt.scatter(y_right_corner[0, 1] * self.image_shape[1], y_right_corner[0, 0] * self.image_shape[0], c="r")
        # plt.scatter(y_corner2[0, 1] * self.image_shape[1], y_corner2[0, 0] * self.image_shape[0], c="r")
        # plt.show()

        eye.pupil = (np.array(
            self.resize_coordinates(y_pupil, shapes_diff, corner_of_frame)) + np.array(eye.pupil)) // 2
        eye.right_corner = (np.array(
            self.resize_coordinates(y_right_corner, shapes_diff, corner_of_frame)) + np.array(eye.right_corner)) // 2
        eye.left_corner = (np.array(
            self.resize_coordinates(y_left_corner, shapes_diff, corner_of_frame)) + np.array(eye.left_corner)) // 2

    def get_eyes_coordinates(self, frame, eye_distances):
        self.get_coordinates_for_one_eye(frame, eye_distances.left_eye, self.eyesnet_left)
        self.get_coordinates_for_one_eye(frame, eye_distances.right_eye, self.eyesnet_right)

        eye_distances.get_distance()


def debug_net():
    eyes_recognizer = EyesRecognizer(
        "/Users/illaria/BSUIR/Diploma/code/PyTorchTry1/eyes_net_left_my_dataset_fixed_photos/epoch_400.pth",
        "/Users/illaria/BSUIR/Diploma/code/PyTorchTry1/eyes_net_right_my_dataset_fixed_photos/epoch_400.pth")

    eye_distances = EyeDistances()
    eye_detector = EyesDetector()
    frame = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/photos_not_moving_pupils_much/result0.jpg', 0)
    results = eye_detector.get_face_mesh_results(frame)
    eye_detector.get_eyes_coordinates(results, frame, eye_distances)
    # plt.imshow(frame)
    # plt.scatter(eye_distances.left_eye.center[0], eye_distances.left_eye.center[1], c="r")
    # plt.scatter(eye_distances.left_eye.inside[0], eye_distances.left_eye.inside[1], cq="r")
    # plt.scatter(eye_distances.left_eye.outside[0], eye_distances.left_eye.outside[1], c="r")
    # plt.show()
    for i in range(1, 20):
        frame = cv2.imread(
            '/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/photos_not_moving_pupils_much/result' + str(i) + '.jpg', 0)
        if i % 6 == 0:
            results = eye_detector.get_face_mesh_results(frame)
            eye_detector.get_eyes_coordinates(results, frame, eye_distances)
        eyes_recognizer.get_eyes_coordinates(frame, eye_distances)
        cv2.drawMarker(frame, eye_distances.left_eye.pupil, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.left_eye.left_corner, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.left_eye.right_corner, (255, 0, 255), markerSize=5)

        cv2.drawMarker(frame, eye_distances.right_eye.pupil, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.right_eye.left_corner, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.right_eye.right_corner, (255, 0, 255), markerSize=5)
        cv2.imshow('frame', frame)
        print(i)
        # print('left distance ' + str(eye_distances.left_distance_avg_x) + '\nright distance ' + str(eye_distances.right_distance_avg_x))
        # print(eye_distances.distance_percentage_x)
        print('angles:')
        print(eye_distances.pupil_angle)
        print(eye_distances.left_corner_angle)
        print(eye_distances.right_corner_angle)

        print('triangle angles')
        print(eye_distances.left_eye.left_angle)
        print(eye_distances.left_eye.right_angle)
        # print_distance(eye_distances)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # cv2.imwrite(
        #     '/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/mynet_myds_fixed_up_down_plus_google/result' + str(
        #         i) + '.jpg', frame)


# main()
debug_net()


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

    corner_of_frame1 = corner1 - np.array([40, 20])  # - np.array([5, 10])
    corner_of_frame2 = corner2 + np.array([30, 20])  # + np.array([5, 10])

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
    annotation_path = os.path.join('/Users/illaria/BSUIR/Diploma/mydataset_only_good_photos/',
                                   patient_name + '/annotationscopy.txt')
    data_folder = os.path.join('/Users/illaria/BSUIR/Diploma/mydataset_only_good_photos/', patient_name)

    annotation = pd.read_csv(annotation_path, sep=" ", header=None)

    points = np.array(annotation.loc[:, list(range(0, 12))])
    print(points)

    filenames = os.listdir(data_folder)
    filenames.sort()
    filenames.remove('annotations.txt')
    filenames.remove('annotationscopy.txt')
    filenames.remove('.DS_Store')
    images = [np.array(ImageEdit.open(os.path.join(data_folder, filename))) for filename in filenames]

    return images, points


def get_my_image_data(images, points):
    for i in range(len(images)):
        signle_image_data = my_image_to_train_data(images[i], points[i])

        if any(stuff is None for stuff in signle_image_data):
            continue

        right_image, right_pupil, right_corner1, right_corner2, left_image, left_pupil, left_corner1, left_corner2 = signle_image_data
        if i == 68:
            plt.imshow(left_image)
            plt.title('left_pupil' + str(i))
            plt.scatter(left_pupil[1] * left_image.shape[1], left_pupil[0] * left_image.shape[0],
                        c="r")  # [0] - y, [1] - x
            plt.scatter(left_corner1[1] * left_image.shape[1], left_corner1[0] * left_image.shape[0], c="r")
            plt.scatter(left_corner2[1] * left_image.shape[1], left_corner2[0] * left_image.shape[0], c="r")
            plt.show()
            plt.imshow(right_image)
            plt.title('right_pupil' + str(i))
            plt.scatter(right_pupil[1] * right_image.shape[1], right_pupil[0] * right_image.shape[0],
                        c="r")  # [0] - y, [1] - x
            plt.scatter(right_corner1[1] * right_image.shape[1], right_corner1[0] * right_image.shape[0], c="r")
            plt.scatter(right_corner2[1] * right_image.shape[1], right_corner2[0] * right_image.shape[0], c="r")
            plt.show()

        if any(right_pupil < 0) or any(left_pupil < 0):
            continue


def create_my_dataset():
    cap = cv2.VideoCapture(0)
    eye_distances = EyeDistances()
    eye_detector = EyesDetector()

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
            results = eye_detector.get_face_mesh_results(frame)
            eye_distances = eye_detector.get_eyes_coordinates(results, frame, eye_distances)
            print(eye_distances.left_eye.bottom[1] - eye_distances.left_eye.top[1])
            if eye_distances.left_eye.bottom[1] - eye_distances.left_eye.top[1] > 14:
                cv2.imwrite('/Users/illaria/BSUIR/Diploma/mydataset/23/' + str(j) + '.jpg', frame)
                annotations.write(
                    str(eye_distances.left_eye.left_corner[1]) + ' ' + str(
                        eye_distances.left_eye.left_corner[0]) + ' ' +
                    str(eye_distances.left_eye.right_corner[1]) + ' ' + str(
                        eye_distances.left_eye.right_corner[0]) + ' ' +
                    str(eye_distances.left_eye.pupil[1]) + ' ' + str(eye_distances.left_eye.pupil[0]) + ' ' +
                    str(eye_distances.right_eye.right_corner[1]) + ' ' + str(
                        eye_distances.right_eye.right_corner[0]) + ' ' +
                    str(eye_distances.right_eye.left_corner[1]) + ' ' + str(
                        eye_distances.right_eye.left_corner[0]) + ' ' +
                    str(eye_distances.right_eye.pupil[1]) + ' ' + str(eye_distances.right_eye.pupil[0]) + '\n')
                j += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    annotations.close()

# create_my_dataset()
