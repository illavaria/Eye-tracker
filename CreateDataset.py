import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import pandas as pd
import os


class ImageEdit():
    default_marker_size = 5
    default_marker_color = (255, 0, 255)
    # default_text_font = cv2.FONT_HERSHEY_SIMPLEX
    default_text_color = (255, 0, 0)
    default_text_thickness = 2
    default_text_font_scale = 1.0

    @staticmethod
    def draw_on_image(image, position, color, marker_size=default_marker_size):
        return cv2.drawMarker(image, position, color, marker_size)

    # @staticmethod
    # def put_text_on_image(image, position, text, font=default_text_font, font_scale=default_text_font_scale,
    #                       color=default_text_color, thickness=default_text_thickness):
    #     return cv2.putText(image, text, position, font, font_scale, color, thickness)

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
