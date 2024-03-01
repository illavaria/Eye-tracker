import cv2
import mediapipe as mp
import numpy as np


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
        frame_h, frame_w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            self.right_eye.center_x, self.right_eye.center_y = int(face_landmarks.landmark[468].x * frame_w), int(face_landmarks.landmark[468].y * frame_h)
            self.left_eye.center_x, self.left_eye.center_y = int(face_landmarks.landmark[473].x * frame_w), int(face_landmarks.landmark[473].y * frame_h)
            self.right_eye.inside_x, self.right_eye.inside_y = int(face_landmarks.landmark[133].x * frame_w), int(face_landmarks.landmark[133].y * frame_h)
            self.left_eye.inside_x, self.left_eye.inside_y = int(face_landmarks.landmark[362].x * frame_w), int(face_landmarks.landmark[362].y * frame_h)
            self.right_eye.outside_x, self.right_eye.outside_y = int(face_landmarks.landmark[33].x * frame_w), int(face_landmarks.landmark[33].y * frame_h)
            self.left_eye.outside_x, self.left_eye.outside_y = int(face_landmarks.landmark[263].x * frame_w), int(face_landmarks.landmark[263].y * frame_h)
            self.right_eye.top_x, self.right_eye.top_y = int(face_landmarks.landmark[159].x * frame_w), int(face_landmarks.landmark[159].y * frame_h)
            self.left_eye.top_x, self.left_eye.top_y= int(face_landmarks.landmark[386].x * frame_w), int(face_landmarks.landmark[386].y * frame_h)
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
    results = eyeDetector.get_face_mesh_results(frame)
    eyeDetector.get_eyes_coordinates(results, frame)
    eye_difference = eyeDetector.left_eye.outside_x- eyeDetector.right_eye.outside_x
    eye_right_x = eyeDetector.right_eye.inside_x - eyeDetector.right_eye.outside_x
    eye_left_x = eyeDetector.left_eye.outside_x - eyeDetector.left_eye.inside_x
    cap.release()
    return eye_difference, eye_right_x, eye_left_x


def main():
    #calibrationData = CalibrationData()
    cap = cv2.VideoCapture(0)
    eyeDetector = EyesDetector()

    eye_difference, eye_right_delta_x, eye_left_delta_x = PreDefine(eyeDetector)
    print(eye_difference)
    i = 0

    while True:
        ret, frame = cap.read()

        # To improve performance
        frame.flags.writeable = False

        results = eyeDetector.get_face_mesh_results(frame)

        # Make it writeable again
        frame.flags.writeable = True

        if not results.multi_face_landmarks:
            continue

        eyeDetector.get_eyes_coordinates(results, frame)
        frame_right_eye = eyeDetector.right_eye.draw(frame)
        result_frame = eyeDetector.left_eye.draw(frame_right_eye)

        #result_frame = cv2.flip(result_frame, 1)
        # Show the image
        cv2.imshow('eyes', result_frame)

        # result_frame = result_frame[eyeDetector.right_eye.outside_y-30:eyeDetector.right_eye.outside_y+30,
        #                eyeDetector.right_eye.outside_x-10:eyeDetector.right_eye.outside_x+10+eye_difference]
        result_frame = result_frame[eyeDetector.left_eye.outside_y - 30:eyeDetector.left_eye.outside_y + 30,
                       eyeDetector.left_eye.outside_x - 20 -eye_left_delta_x:eyeDetector.left_eye.outside_x + 15]
        cv2.imshow('eyes difference', result_frame)
        cv2.imwrite('/Users/illaria/BSUIR/Diploma/code/MediaPipeTry1/left_eye3/result' + str(i) + '.jpg', result_frame)
        i += 1
        if i == 20:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
