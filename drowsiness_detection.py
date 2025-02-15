from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_open_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])  # distance between upper and lower lip
    B = distance.euclidean(mouth[0], mouth[6])  # distance between mouth corners
    mor = A / B
    return mor

def nose_to_lip_ratio(nose, mouth):
    C = distance.euclidean(nose[3], mouth[3])  # distance between nose tip and upper lip
    D = distance.euclidean(nose[0], mouth[0])  # distance between nose corners and mouth corners
    nl_ratio = C / D
    return nl_ratio

def main():
    try:
        thresh = 0.25
        frame_check = 25
        detect = dlib.get_frontal_face_detector()
        predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["nose"]

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        mixer.init()
        mixer.music.load("music.wav")

        flag = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot receive frame")
                break

            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detect(gray, 0)
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]
                nose = shape[nStart:nEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                mor = mouth_open_ratio(mouth)
                nl_ratio = nose_to_lip_ratio(nose, mouth)

                # Draw markings for EAR
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


                # Draw markings for MOR using convex hull
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 2)

                # Draw markings for NL Ratio
                cv2.line(frame, (nose[3][0], nose[3][1]), (mouth[3][0], mouth[3][1]), (255, 0, 0), 1)
                cv2.line(frame, (nose[0][0], nose[0][1]), (mouth[0][0], mouth[0][1]), (255, 0, 0), 1)

                if ear < thresh or mor > 0.5 or nl_ratio > 0.7:
                    flag += 1
                    print(flag)
                    if flag >= frame_check:
                        cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mixer.music.play()
                else:
                    flag = 0

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except Exception as e:
        print("An error occurred: ", str(e))

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
