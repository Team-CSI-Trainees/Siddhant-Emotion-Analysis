import cv2 as cv
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    result = DeepFace.analyze(frame, actions=["age", "gender", "emotion", "race"])
    print(result)
    detector = MTCNN()
    faces = detector.detect_faces(frame)  # result
    font = cv.FONT_HERSHEY_SIMPLEX
    # to draw faces on image
    for output in faces:
        x, y, w, h = output['box']
        x1, y1 = x + w, y + h
        cv.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

        ##TEXT##

        #emotion#
        cv.putText(frame, result['dominant_emotion'], (x+25, y+25), font, 1, (0, 255, 0), 2, cv.LINE_AA)

        #age#
        cv.putText(frame, str(result['age']), (x, y), font, 1, (0, 255, 0), 2, cv.LINE_AA)

        #gender#
        cv.putText(frame, result['gender'], (x-25, y-25), font, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow("Orignal Video", frame)

    if cv.waitKey(50) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()