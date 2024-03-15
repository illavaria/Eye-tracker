import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time


class Eye:
    center_x, center_y = 0, 0
    inside_x, inside_y = 0, 0
    outside_x, outside_y = 0, 0
    top_x, top_y = 0, 0
    bottom_x, bottom_y = 0, 0
    
    def draw(self, frame):
        cv2.drawMarker(frame, (self.center_x, self.center_y), (0, 255, 0), markerSize=5)
        cv2.drawMarker(frame, (self.inside_x, self.inside_y), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.outside_x, self.outside_y), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.top_x, self.top_y), (255, 0, 255), markerSize=5)
        cv2.drawMarker(frame, (self.bottom_x, self.bottom_y), (255, 0, 255), markerSize=5)
        return frame
    

class EyesDetector:
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                refine_landmarks=True,
                                                max_num_faces=2,
                                                min_detection_confidence=0.5)
    left_eye = Eye()
    right_eye = Eye()

    def get_face_mesh_results(self, frame):
        return self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_eyes_coordinates(self, results, frame):
        frame_h, frame_w = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            self.right_eye.center_x, self.right_eye.center_y = int(face_landmarks.landmark[468].x * frame_w), int(face_landmarks.landmark[468].y * frame_h)
            self.left_eye.center_x, self.left_eye.center_y = int(face_landmarks.landmark[473].x * frame_w), int(face_landmarks.landmark[473].y * frame_h)
            self.right_eye.inside_x, self.right_eye.inside_y = int(face_landmarks.landmark[133].x * frame_w), int(face_landmarks.landmark[133].y * frame_h)
            self.left_eye.inside_x, self.left_eye.inside_y = int(face_landmarks.landmark[362].x * frame_w), int(face_landmarks.landmark[362].y * frame_h)
            self.right_eye.outside_x, self.right_eye.outside_y = int(face_landmarks.landmark[33].x * frame_w), int(face_landmarks.landmark[33].y * frame_h)
            self.left_eye.outside_x, self.left_eye.outside_y = int(face_landmarks.landmark[263].x * frame_w), int(face_landmarks.landmark[263].y * frame_h)
            self.right_eye.top_x, self.right_eye.top_y = int(face_landmarks.landmark[159].x * frame_w), int(face_landmarks.landmark[159].y * frame_h)
            self.left_eye.top_x, self.left_eye.top_y = int(face_landmarks.landmark[386].x * frame_w), int(face_landmarks.landmark[386].y * frame_h)
            self.right_eye.bottom_x, self.right_eye.bottom_y = int(face_landmarks.landmark[145].x * frame_w), int(face_landmarks.landmark[145].y * frame_h)
            self.left_eye.bottom_x, self.left_eye.bottom_y = int(face_landmarks.landmark[374].x * frame_w), int(face_landmarks.landmark[374].y * frame_h)

            # # Calculate left eye scores (x,y)
            # if (inside_x - outside_x) != 0:
            #     lx_score = (lcx - outside_x) / (inside_x - outside_x)
            #     if abs(lx_score - self.last_x) < .3:
            #         cv2.putText(frame, "x: {:.02}".format(lx_score), (lcx, lcy - 30), cv2.FONT_HERSHEY_SIMPLEX, .25,
            #                     (255, 255, 255), 1)
            #     last_x = lx_score
            # 
            # if (bottom - top) != 0:
            #     ly_score = (lcy - top) / (bottom - top)
            #     if abs(ly_score - self.last_y) < .3:
            #         cv2.putText(frame, "y: {:.02}".format(ly_score), (lcx, lcy - 20), cv2.FONT_HERSHEY_SIMPLEX, .25,
            #                     (255, 255, 255), 1)
            #     last_y = ly_score

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
            results = eyeDetector.get_face_mesh_results(frame)
            if not results.multi_face_landmarks:
                continue
            eyeDetector.get_eyes_coordinates(results, frame)
            frame_right_eye = eyeDetector.right_eye.draw(frame)
            result_frame = eyeDetector.left_eye.draw(frame_right_eye)
            result_frame = cv2.flip(result_frame, 1)
            cv2.imshow('calibrate eyes', result_frame)

            if cv2.waitKey(1) & 0xFF == ord('a'):
                lct_left_eye, lct_right_eye = eyeDetector.left_eye, eyeDetector.right_eye
                cv2.imwrite('result' + str(i) + '.jpg', result_frame)
                i += 1
                # cap.release()
                #cv2.destroyWindow('calibrate eyes')
                #cv2.destroyAllWindows()
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
    eye_difference = eyeDetector.left_eye.outside_x- eyeDetector.right_eye.outside_x
    eye_right_x = eyeDetector.right_eye.inside_x - eyeDetector.right_eye.outside_x
    eye_left_x = eyeDetector.left_eye.outside_x - eyeDetector.left_eye.inside_x
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
    n = 17
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
    x0, y0 = left_eye.outside_x, left_eye.outside_y
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

# main()
# Contrast()
# Fixate()
Test()

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
