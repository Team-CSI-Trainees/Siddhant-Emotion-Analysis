import cv2 as cv
from mtcnn.mtcnn import MTCNN
from  deepface import DeepFace

capture = cv.VideoCapture(0)

while True:
    ret,frame = capture.read()
    result = DeepFace.analyze(frame, actions = ['emotion', 'age', 'gender'])
    
    detector = MTCNN()
    faces = detector.detect_faces(frame)# result
    #to draw faces on image
    for output in faces:
        x, y, w, h = output['box']
        x1, y1 = x + w, y + h
        cv.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
    font = cv.FONT_HERSHEY_SIMPLEX
    #dominant_emotion = result['dominant_emotion']
    cv.putText(frame,
              result['dominant_emotion'],
              (50, 50), font, 3,
              (0, 0, 255),
              2, cv.LINE_4)
    
    #cv.putText(frame,
    #         result['age'],
    #         (70, 70), font, 3,
    #         (0, 0, 255),
    #         2, cv.LINE_4)
    
    cv.putText(frame,
              result['gender'],
              (90, 90), font, 3,
              (0, 0, 255),
              2, cv.LINE_4)    
    
    cv.imshow("Orignal Video", frame)
    
    if cv.waitKey(50) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()