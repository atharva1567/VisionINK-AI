import cv2

FACE_MASK_MODE = "PIXELATE"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def apply_face_mask(frame, mode=FACE_MASK_MODE):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        if mode == "BLUR":
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(face, (99, 99), 30)

        elif mode == "BLACK":
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)

        else:
            small = cv2.resize(face, (16, 16))
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y+h, x:x+w] = pixelated

    return frame