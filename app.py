import cv2 as cv
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace
from flask import Flask, render_template, Response

app = Flask(__name__)

capture = cv.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = capture.read()
        if not success:
            break
        else:
            result = DeepFace.analyze(
                frame, actions=["age", "gender", "emotion", "race"])
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
                cv.putText(frame,
                           result['dominant_emotion'],
                           (x, y+50),
                           font, 1,
                           (0, 255, 0), 2,
                           cv.LINE_8)

                #age#
                cv.putText(frame,
                           str(result['age']),
                           (x, y),
                           font, 1,
                           (0, 255, 0), 2,
                           cv.LINE_8)

                #gender#
                cv.putText(frame,
                           result['gender'],
                           (x, y+25),
                           font, 1,
                           (0, 255, 0), 2,
                           cv.LINE_8)

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
