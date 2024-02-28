import cv2
import mediapipe as mp
import numpy as np


class EyesDetector:
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                refine_landmarks=True,
                                                max_num_faces=2,
                                                min_detection_confidence=0.5)

    last_x = 0
    last_y = 0

    def GetFaceMeshResults(self, frame):
        return self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def GetEyesCoordinates(self, results, frame):
        frame_h, frame_w, _ = frame.shape
        for face_landmarks in results.multi_face_landmarks:

            # Eye center landmarks
            rcx, rcy, lcx, lcy = 0, 0, 0, 0

            # Eye corner landmarks
            inside_x, outside_x = 0, 0

            # Eye top/bottom landmarks
            top, bottom = 0, 0

            # Iterate over all landmark
            for idx, lm in enumerate(face_landmarks.landmark):
                # !!! ONLY WORKS FOR ONE EYE, FIXED FOR IN HEADPOSE.PY ITERATION !!!

                # Right iris center landmark
                if idx == 468:
                    rcx, rcy = int(lm.x * frame_w), int(lm.y * frame_h)
                    cv2.drawMarker(frame, (rcx, rcy), (0, 255, 0), markerSize=5)

                # Left iris center landmark
                if idx == 473:
                    lcx, lcy = int(lm.x * frame_w), int(lm.y * frame_h)
                    cv2.drawMarker(frame, (lcx, lcy), (0, 255, 0), markerSize=5)

                    # Eye inside corner landmark
                if idx == 133 or idx == 362: # for left eye 155 is a little bit to the left
                    inside_x, inside_y = int(lm.x * frame_w), int(lm.y * frame_h)
                    cv2.drawMarker(frame, (inside_x, inside_y), (255, 0, 255), markerSize=5)

                    # Eye outside corner landmark
                if idx == 33 or idx == 263: # for left eye 130 is like the corner of skin, 33 of purpil, 466 for right
                    outside_x, outside_y = int(lm.x * frame_w), int(lm.y * frame_h)
                    cv2.drawMarker(frame, (outside_x, outside_y), (255, 0, 255), markerSize=5)

                # Left/right Top eye landmark
                if idx == 159 or idx == 386:
                    top_x, top = int(lm.x * frame_w), int(lm.y * frame_h)
                    cv2.drawMarker(frame, (top_x, top), (255, 0, 255), markerSize=5)

                    # Left/right Bottom eye landmark
                if idx == 145 or idx == 374:
                    bottom_x, bottom = int(lm.x * frame_w), int(lm.y * frame_h)
                    cv2.drawMarker(frame, (bottom_x, bottom), (255, 0, 255), markerSize=5)

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
        return frame



def main():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    eyeDetector = EyesDetector()

    while True:
        ret, frame = cap.read()

        # To improve performance
        frame.flags.writeable = False

        results = eyeDetector.GetFaceMeshResults(frame)

        # Make it writeable again
        frame.flags.writeable = True

        frame_h, frame_w, _ = frame.shape

        if not results.multi_face_landmarks:
            continue

        result_frame = eyeDetector.GetEyesCoordinates(results, frame)

        result_frame = cv2.flip(result_frame, 1)
        # Show the image
        cv2.imshow('eyes', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
