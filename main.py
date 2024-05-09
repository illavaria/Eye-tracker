import enum
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
    def __init__(self, camera_number):
        self.camera_number = camera_number
        # add checks
        self.capture = None

    def is_capturing(self):
        return False if self.capture is None else True

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
        self.capture = None

    @staticmethod
    def write_frame(frame, address_for_write):
        cv2.imwrite(address_for_write, frame)

    @staticmethod
    def read_frame_from_file(address_for_read):
        return cv2.imread(address_for_read)

    @staticmethod
    def get_camera_name_from_port(port):
        cap = cv2.VideoCapture(port)
        if cap.isOpened():
            _, frame = cap.read()
            cap.release()
            return f"Port {port}: {frame.shape[1]}x{frame.shape[0]}"
        else:
            return "No camera"

    @staticmethod
    def get_camera_names(ports):
        result = []
        for port in ports:
            result.append(Camera.get_camera_name_from_port(port))
        return result

    @staticmethod
    def list_ports():
        index = 0
        ports = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                ports.append(index)
            cap.release()
            index += 1
        return ports

class ImageEdit():
    default_marker_size = 5
    default_marker_color = (255, 0, 255)
    default_text_font = cv2.FONT_HERSHEY_SIMPLEX
    default_text_color = (255, 0, 0)
    default_text_thickness = 2
    default_text_font_scale = 1.0

    @staticmethod
    def draw_on_image(image, position, color, marker_size=default_marker_size):
        return cv2.drawMarker(image, position, color, marker_size)

    @staticmethod
    def put_text_on_image(image, position, text, font=default_text_font, font_scale=default_text_font_scale,
                          color=default_text_color, thickness=default_text_thickness):
        return cv2.putText(image, text, position, font, font_scale, color, thickness)

    @staticmethod
    def split_image(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, _, frame = cv2.split(hsv)
        return frame

    @staticmethod
    def image_to_RGB(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def resize_image(image, image_shape):
        return cv2.resize(image, image_shape, interpolation=cv2.INTER_AREA)

    @staticmethod
    def flip_image(image):
        return cv2.flip(image, 1)





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


class ImagesLoader():
    def __init__(self, folder, non_images_files):
        self.folder_path = folder
        self.non_images_files = non_images_files

    def load_images(self):
        files = os.listdir(self.folder_path)
        for f in self.non_images_files:
            files.remove(f)

        files.sort()
        images = [np.array(Image.open(os.path.join(self.folder_path, file))) for file in files]
        return images

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
        # cv2.drawMarker(frame, (self.top[0], self.top[1]), (255, 0, 255), markerSize=5)
        # cv2.drawMarker(frame, (self.bottom[0], self.bottom[1]), (255, 0, 255), markerSize=5)
        return frame


class EyeDistances:
    def __init__(self):
        self.left_eye = Eye()
        self.right_eye = Eye()
        self.standard_x = 0.0
        self.left_distance_avg_x, self.right_distance_avg_x = 0.0, 0.0
        self.distance_percentage_x = 0.0  # left/right
        self.pupils_line_angle = 0.0
        self.right_corner_line_angle = 0.0
        self.left_corner_line_angle = 0.0
        self.right_angle_avg = 0.0
        self.left_angle_avg = 0.0
        self.angle_avg = 0.0

    @staticmethod
    def get_angle(dot1, dot2):
        return math.atan((dot1[1] - dot2[1]) / (dot1[0] - dot2[0])) * 180 / math.pi

    @staticmethod
    def calculate_angle(dot1, dot2, middle_dot):
        # Векторы прямых
        vec1 = (middle_dot[0] - dot2[0], middle_dot[1] - dot2[1])
        vec2 = (middle_dot[0] - dot1[0], middle_dot[1] - dot1[1])

        # Скалярное произведение векторов
        dot_product = vec2[0] * vec1[0] + vec2[1] * vec1[1]

        # Нормы (длины) векторов
        norm_length1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        norm_length2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

        cos_theta = dot_product / (norm_length1 * norm_length2)
        theta_radians = math.acos(cos_theta)
        theta_degrees = math.degrees(theta_radians)

        return theta_degrees

    def get_distances_for_one_eye(self, eye):
        eye.left_distance = [(eye.left_corner[0] - eye.pupil[0]) / self.standard_x,
                             (eye.left_corner[1] - eye.pupil[1]) / self.standard_x]
        eye.right_distance = [(eye.pupil[0] - eye.right_corner[0]) / self.standard_x,
                              (eye.pupil[1] - eye.right_corner[1] / self.standard_x)]
        # rclc_angle = self.get_angle(eye.right_corner, eye.left_corner)
        # lcp = self.get_angle(eye.pupil, eye.left_corner)
        # eye.left_angle = lcp + rclc_angle
        eye.left_angle = self.calculate_angle(eye.pupil, eye.right_corner, eye.left_corner)
        # rcp = 0 - self.get_angle(eye.right_corner, eye.pupil)
        # eye.right_angle = rcp - rclc_angle
        eye.right_angle = self.calculate_angle(eye.pupil, eye.left_corner, eye.right_corner)
        # big_angle = 180 - rcp - lcp
        # print(big_angle)

    def get_distance(self):
        self.standard_x = self.left_eye.pupil[0] - self.right_eye.pupil[0]

        self.pupils_line_angle = self.get_angle(self.left_eye.pupil, self.right_eye.pupil)
        self.right_corner_line_angle = self.get_angle(self.left_eye.right_corner, self.right_eye.right_corner)
        self.left_corner_line_angle = self.get_angle(self.left_eye.left_corner, self.right_eye.left_corner)

        self.get_distances_for_one_eye(self.left_eye)
        self.get_distances_for_one_eye(self.right_eye)

        self.left_distance_avg_x = (self.left_eye.left_distance[0] + self.right_eye.left_distance[0]) / 2
        self.right_distance_avg_x = (self.left_eye.right_distance[0] + self.right_eye.right_distance[0]) / 2

        self.distance_percentage_x = self.left_distance_avg_x / self.right_distance_avg_x

        self.left_angle_avg = (self.left_eye.left_angle + self.right_eye.left_angle) / 2
        self.right_angle_avg = (self.left_eye.right_angle + self.right_eye.right_angle) / 2
        self.angle_avg = (self.left_angle_avg + self.right_angle_avg) / 2

    def draw(self, frame):
        frame = self.left_eye.draw(frame)
        return self.right_eye.draw(frame)


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

class CheatingDetection:
    def __init__(self):
        self.lct = EyeDistances()
        self.tm = EyeDistances()
        self.rct = EyeDistances()
        self.rm = EyeDistances()
        self.lm = EyeDistances()
        self.middle = EyeDistances()
        self.lcb = EyeDistances()
        self.bm = EyeDistances()
        self.rcb = EyeDistances()

        self.left_border_percentage_x_avg = 0.0
        self.right_border_percentage_x_avg = 0.0
        self.left_border_angle_diff_avg = 0.0
        self.right_border_angle_diff_avg = 0.0
        self.up_border_angle_diff_avg = 0.0
        self.down_border_angle_diff_avg = 0.0


    def predict(self, eye_distances):
        if eye_distances.distance_percentage_x + 0.13 < self.left_border_percentage_x_avg :
            return True, Direction.left
        elif eye_distances.distance_percentage_x > self.right_border_percentage_x_avg + 0.14:
            return True, Direction.right
        elif (eye_distances.angle_avg / eye_distances.standard_x) - self.up_border_angle_diff_avg > 0.014:
            return True, Direction.up
        elif self.down_border_angle_diff_avg - (eye_distances.angle_avg / eye_distances.standard_x) > 0.015:
            return True, Direction.down
        else:
            return False, Direction.forward

    def calculate_borders(self):
        self.left_border_percentage_x_avg = (self.lct.distance_percentage_x + self.lcb.distance_percentage_x +
                                             self.lm.distance_percentage_x) / 3
        self.right_border_percentage_x_avg = (self.rct.distance_percentage_x + self.rcb.distance_percentage_x +
                                              self.rm.distance_percentage_x) / 3
        self.left_border_angle_diff_avg = ((self.lct.left_angle_avg - self.lct.right_angle_avg) / self.lct.standard_x +
                                           (self.lcb.left_angle_avg - self.lcb.right_angle_avg) / self.lcb.standard_x +
                                           (self.lm.left_angle_avg - self.lm.right_angle_avg) / self.lm.standard_x) / 3
        self.right_border_angle_diff_avg = ((self.rct.right_angle_avg - self.rct.left_angle_avg) / self.rct.standard_x +
                                            (self.rcb.right_angle_avg - self.rcb.left_angle_avg) / self.rcb.standard_x+
                                            (self.rm.right_angle_avg - self.rm.left_angle_avg) / self.rm.standard_x)
        self.up_border_angle_diff_avg = (self.lct.angle_avg / self.lct.standard_x +
                                         self.tm.angle_avg / self.tm.standard_x +
                                         self.rct.angle_avg / self.rct.standard_x) / 3
        self.down_border_angle_diff_avg = (self.lcb.angle_avg / self.lcb.standard_x +
                                           self.bm.angle_avg / self.bm.standard_x +
                                           self.rcb.angle_avg / self.rcb.standard_x) / 3
        # self.up_border_angle_diff_avg = (self.lct.angle_avg +
        #                                  self.tm.angle_avg +
        #                                  self.rct.angle_avg ) / 3
        # self.down_border_angle_diff_avg = (self.lcb.angle_avg +
        #                                    self.bm.angle_avg +
        #                                    self.rcb.angle_avg) /3



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

    print('left eye:  ', eye_distances.left_eye.left_distance[0],
          eye_distances.left_eye.right_distance[0])
    print('right eye: ', eye_distances.right_eye.left_distance[0],
          eye_distances.right_eye.right_distance[0])


def calculate_distance_for_eyesnet(x1, y1, x2, y2):
    return int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)


class EyesRecognizer:
    def __init__(self):
        self.eyesnet_left = EyesNet()
        self.eyesnet_right = EyesNet()
        self.is_loaded = False
        self.image_shape = (16, 32)

    def load_state(self, path_left, path_right):
        self.eyesnet_left.load_state_dict(torch.load(path_left))
        self.eyesnet_right.load_state_dict(torch.load(path_right))
        self.is_loaded = True

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        return int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

    def get_eye_image(self, image, corner1, corner2):
        line_len = self.calculate_distance(*corner1, *corner2)
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


class Direction (enum.Enum):
    forward = 'forward'
    left = 'left'
    right = 'right'
    up = 'up'
    down = 'down'


class GazeDirectionPrediction:
    def __init__(self):
        self.left_threshold = 0.8
        self.right_threshold = 1.25
        self.up_threshold = 20.0
        self.down_threshold = 3.0

    def predict(self, eye_distances):
        if abs(eye_distances.left_angle_avg - eye_distances.right_angle_avg) > 3:
            if eye_distances.distance_percentage_x < self.left_threshold:
                return Direction.left
            elif eye_distances.distance_percentage_x > self.right_threshold:
                return Direction.right
        else:
            if eye_distances.angle_avg > 12:
                return Direction.up
            elif eye_distances.angle_avg < 7:
                return Direction.down
        return Direction.forward

def debug_net():
    eyes_recognizer = EyesRecognizer()
    eyes_recognizer.load_state(
        "/Users/illaria/BSUIR/Diploma/code/PyTorchTry1/eyes_net_left_my_dataset_fixed_photos/epoch_400.pth",
        "/Users/illaria/BSUIR/Diploma/code/PyTorchTry1/eyes_net_right_my_dataset_fixed_photos/epoch_400.pth")
    print(eyes_recognizer.is_loaded)
    eye_distances = EyeDistances()
    eye_detector = EyesDetector()
    gaze_prediction = GazeDirectionPrediction()
    image_loader = ImagesLoader('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/all_photos/photos_up_down', [])
    images = image_loader.load_images()
    frame = images[0]
    # frame = cv2.imread('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/photos_not_moving_pupils_much/result0.jpg', 0)
    results = eye_detector.get_face_mesh_results(frame)
    eye_detector.get_eyes_coordinates(results, frame, eye_distances)
    # plt.imshow(frame)
    # plt.scatter(eye_distances.left_eye.center[0], eye_distances.left_eye.center[1], c="r")
    # plt.scatter(eye_distances.left_eye.inside[0], eye_distances.left_eye.inside[1], cq="r")
    # plt.scatter(eye_distances.left_eye.outside[0], eye_distances.left_eye.outside[1], c="r")
    # plt.show()
    for i in range(1, 20):
        frame = images[i]
        # if i % 3 == 0:
        results = eye_detector.get_face_mesh_results(frame)
        eye_detector.get_eyes_coordinates(results, frame, eye_distances)
        # start_time = time.time()
        eyes_recognizer.get_eyes_coordinates(frame, eye_distances)
        # end_time = time.time()
        # print(end_time - start_time)
        cv2.drawMarker(frame, eye_distances.left_eye.pupil, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.left_eye.left_corner, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.left_eye.right_corner, (255, 0, 255), markerSize=5)

        cv2.drawMarker(frame, eye_distances.right_eye.pupil, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.right_eye.left_corner, (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, eye_distances.right_eye.right_corner, (255, 0, 255), markerSize=5)
        cv2.imshow('frame', frame)
        print(i)
        print('left distance ' + str(eye_distances.left_distance_avg_x) + '\nright distance ' + str(eye_distances.right_distance_avg_x))
        # print(eye_distances.distance_percentage_x)
        print('angles:')
        print(eye_distances.pupils_line_angle)
        print(eye_distances.left_corner_line_angle)
        print(eye_distances.right_corner_line_angle)

        print('triangle angles')
        print('left:')
        print(eye_distances.left_eye.left_angle)
        print(eye_distances.left_eye.right_angle)
        print('right:')
        print(eye_distances.right_eye.left_angle)
        print(eye_distances.right_eye.right_angle)
        print('avg left and right angles')
        print(eye_distances.left_angle_avg)
        print(eye_distances.right_angle_avg)
        print(' ')
        print(gaze_prediction.predict(eye_distances))
        print_distance(eye_distances)

        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        cv2.imwrite(
            '/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/all_photos/mynet_myds_up_down/result' + str(
                i) + '.jpg', frame)


# main()
# debug_net()


# cap = cv2.VideoCapture(0)
# while True:
#     start_time = time.time()
#     ret, frame = cap.read()
#     end_time = time.time()
#     print(end_time - start_time)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, frame = cv2.threshold(frame, 127, 255, 0)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # _, _, frame = cv2.split(hsv)
    # cv2.imshow('eyes', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

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
    image_loader = ImagesLoader(data_folder, ['annotationscopy.txt', 'annotations.txt', '.DS_Store'])
    images = image_loader.load_images()

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

# images, pupils = load_my_image_data('15')
# get_my_image_data(images, pupils)

def create_my_dataset():
    cap = cv2.VideoCapture(0)
    eye_distances = EyeDistances()
    eye_detector = EyesDetector()

    ret, frame = cap.read()
    ret, frame = cap.read()
    annotations = open('/Users/illaria/BSUIR/Diploma/mydataset/24/annotations.txt', 'x')
    j = 0
    for i in range(200):
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, _, frame = cv2.split(hsv)
        cv2.imshow('eyes', frame)
        if i % 2 == 0:
            results = eye_detector.get_face_mesh_results(frame)
            eye_detector.get_eyes_coordinates(results, frame, eye_distances)
            print(eye_distances.left_eye.bottom[1] - eye_distances.left_eye.top[1])
            if eye_distances.left_eye.bottom[1] - eye_distances.left_eye.top[1] > 14:
                cv2.imwrite('/Users/illaria/BSUIR/Diploma/mydataset/24/' + str(j) + '.jpg', frame)
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
