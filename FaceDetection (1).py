import cv2 
import dlib 
import imutils
from flask import Flask

app = Flask(__name__)

def convert_and_trim_bb(image, rect):
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	w = endX - startX
	h = endY - startY
	return (startX, startY, w, h)

@app.route("/", methods=["GET"])
def runFaceDetection():
    faceDetector = dlib.get_frontal_face_detector()
    cam = cv2.VideoCapture(0)
    while True:
        _, frame = cam.read()
        frame = imutils.resize(frame, width=640)
        faces = faceDetector(frame)
        if(len(faces) > 0):
            cv2.putText(img=frame, text="Hi human being", org=(5,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
        else:
            cv2.putText(img=frame, text="Not A Human Being", org=(5,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
        boxes = [convert_and_trim_bb(frame, r) for r in faces]
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)    
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    return {'success': True}

if __name__ == '__main__':
    app.run(host="localhost", port=8080, debug=True)