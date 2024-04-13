import os
import cv2
import numpy as np

import pandas as pd
from PIL import  Image
from matplotlib import pyplot as plt

click_count = 0
coordinates_list = []

# Функция для обработки событий мыши
def click_event(event, x, y, flags, param):
    global click_count, coordinates_list

    if event == cv2.EVENT_LBUTTONDOWN:
        click_count += 1
        if click_count <= 3:
            coordinates_list.append((y + corner_of_frame_left[0], x + corner_of_frame_left[1]))  # Замена x и y для сохранения в формате "y x"
            cv2.circle(img1, (x, y), 1, (0, 255, 0), -1)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1600, 800)
            cv2.imshow('image', img1)
        else:
            coordinates_list.append((y + corner_of_frame_right[0], x + corner_of_frame_right[1]))  # Замена x и y для сохранения в формате "y x"
            cv2.circle(img2, (x, y), 1, (0, 255, 0), -1)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 1600, 800)
            cv2.imshow('image', img2)

        if click_count == 6:
            cv2.destroyWindow('image')
            click_count = 0
            save_coordinates(coordinates_list)
            coordinates_list = []


# Функция для сохранения координат в текстовый файл
def save_coordinates(coordinates):
    with open('/Users/illaria/BSUIR/Diploma/mydataset_only_good_photos/15/annotationscopy.txt', 'a') as file:
        for i in range(0, len(coordinates)):
            file.write(f'{coordinates[i][0]} {coordinates[i][1]} ')  # Запись координат в формате "y x"
        file.write('\n')

def load_my_image_data(patient_name):

    annotation_path = os.path.join('/Users/illaria/BSUIR/Diploma/mydataset_only_good_photos/', patient_name + '/annotations.txt')
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

corner_of_frame_left = np.array([5, 10])
corner_of_frame_right = np.array([5, 10])

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

    return sub_image, pupil_new, corner1_new, corner2_new, corner_of_frame1

def my_image_to_train_data(image, points):
    global  corner_of_frame_left, corner_of_frame_right
    eye_right_c1 = points[6:8]
    eye_right_c2 = points[8:10]
    eye_right_pupil = points[10:12]
    eye_right_c1 = eye_right_c1[::-1]
    eye_right_c2 = eye_right_c2[::-1]
    eye_right_pupil = eye_right_pupil[::-1]

    right_image, right_pupil, right_corner1, right_corner2, corner_of_frame_right = handle_eye(image, eye_right_c1, eye_right_c2,
                                                                        eye_right_pupil)

    eye_left_c1 = points[0:2]
    eye_left_c2 = points[2:4]
    eye_left_pupil = points[4:6]
    eye_left_c1 = eye_left_c1[::-1]
    eye_left_c2 = eye_left_c2[::-1]
    eye_left_pupil = eye_left_pupil[::-1]

    left_image, left_pupil, left_corner1, left_corner2, corner_of_frame_left = handle_eye(image, eye_left_c1, eye_left_c2, eye_left_pupil)

    return right_image, right_pupil, right_corner1, right_corner2, left_image, left_pupil, left_corner1, left_corner2

def get_my_image_data(images, points):
    global img1, img2
    for i in range(len(images)):
        signle_image_data = my_image_to_train_data(images[i], points[i])
        print(i)

        if any(stuff is None for stuff in signle_image_data):
            continue

        right_image, right_pupil, right_corner1, right_corner2, left_image, left_pupil, left_corner1, left_corner2 = signle_image_data
        img1 = left_image
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('image', 1600, 800)
        cv2.imshow('image', img1)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1600, 800)
        cv2.setMouseCallback('image', click_event)

        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        img2 = right_image
        cv2.setMouseCallback('image', click_event)
        if click_count == 3:
            cv2.imshow('image', img2)
        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        # img2 = right_image
        # cv2.imshow('image', img2)
        # cv2.destroyWindow('right_image')
        # while True:
        #     key = cv2.waitKey(1)
        #     if key == ord('q'):
        #         break


        if any(right_pupil < 0) or any(left_pupil < 0):
            continue

def get_fine_images(images, points, pattern):
    j = 0
    for i in range(len(images)):
        signle_image_data = my_image_to_train_data(images[i], points[i])
        print(i)

        if any(stuff is None for stuff in signle_image_data):
            continue

        right_image, right_pupil, right_corner1, right_corner2, left_image, left_pupil, left_corner1, left_corner2 = signle_image_data
        cv2.imshow('left_image', left_image)
        cv2.namedWindow('left_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('left_image', 800, 400)
        cv2.imshow('right_image', right_image)
        cv2.namedWindow('right_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('right_image', 800, 400)
        while True:
            key = cv2.waitKey(1)
            if key == ord('w'):
                cv2.imwrite('/Users/illaria/BSUIR/Diploma/mydataset_only_good_photos/' + str(pattern) + '/' + str(j)+'.jpg', images[i])
                with open('/Users/illaria/BSUIR/Diploma/mydataset_only_good_photos/' + str(pattern) + '/annotationscopy.txt', 'a') as file:
                    file.write(str(points[i][0:12]).removeprefix('[ ').removesuffix(']').replace('  ', ' ') + '\n')
                j += 1
                break
            if key == ord('q'):
                break




pattern = '15'
images, points = load_my_image_data(pattern)
img1 = images[0]
img2 = images[1]
get_my_image_data(images, points)
# get_fine_images(images, points, pattern)



# Проходим по каждому файлу
