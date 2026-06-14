import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# =========================
# AirTeach AI - Virtual Whiteboard + Privacy Face Mask
#
# Controls:
# - Index finger up: draw
# - Index + middle up: move/select without drawing
# - Pinch index + thumb: eraser
# - Click top toolbar with index finger to change color/clear/save
# - Press m to switch face mask mode
# - Press c to clear
# - Press s to save
# - Press q to quit
# =========================

CAMERA_INDEX = 1
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DRAW_THICKNESS = 8
ERASER_THICKNESS = 45
TOOLBAR_HEIGHT = 90
SAVE_FOLDER = "saved_boards"

# Face mask options: "PIXELATE", "BLUR", "BLACK"
FACE_MASK_MODE = "PIXELATE"

COLORS = {
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "WHITE": (255, 255, 255),
}

TOOLS = [
    ("BLUE", COLORS["BLUE"]),
    ("GREEN", COLORS["GREEN"]),
    ("RED", COLORS["RED"]),
    ("YELLOW", COLORS["YELLOW"]),
    ("WHITE", COLORS["WHITE"]),
    ("ERASER", (60, 60, 60)),
    ("CLEAR", (40, 40, 40)),
    ("SAVE", (90, 90, 90)),
]

os.makedirs(SAVE_FOLDER, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def apply_face_mask(frame, mode="PIXELATE"):
    """Anonymizes faces in the frame using pixelate, blur, or black box."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
    )

    for (x, y, w, h) in faces:
        # Add padding so the mask covers more of the face/head
        pad_x = int(w * 0.18)
        pad_y = int(h * 0.25)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)

        face_region = frame[y1:y2, x1:x2]

        if face_region.size == 0:
            continue

        if mode == "BLUR":
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_region, (99, 99), 30)

        elif mode == "BLACK":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

        else:  # PIXELATE
            small = cv2.resize(face_region, (16, 16), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(
                small,
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST,
            )
            frame[y1:y2, x1:x2] = pixelated

    return frame


def fingers_up(hand_landmarks):
    """Returns [thumb, index, middle, ring, pinky], where 1 means finger is up."""
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb: compare x positions. Works best for right hand.
    fingers.append(1 if lm[4].x < lm[3].x else 0)

    # Other fingers: fingertip y is above pip joint y
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)

    return fingers


def distance(p1, p2):
    return int(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def draw_toolbar(frame, selected_tool):
    h, w, _ = frame.shape
    button_w = w // len(TOOLS)

    for i, (name, color) in enumerate(TOOLS):
        x1 = i * button_w
        x2 = x1 + button_w

        cv2.rectangle(frame, (x1, 0), (x2, TOOLBAR_HEIGHT), color, -1)

        border_color = (255, 255, 255) if name == selected_tool else (0, 0, 0)
        border_thick = 4 if name == selected_tool else 2
        cv2.rectangle(frame, (x1, 0), (x2, TOOLBAR_HEIGHT), border_color, border_thick)

        text_color = (0, 0, 0) if name in ["YELLOW", "WHITE", "GREEN"] else (255, 255, 255)
        cv2.putText(
            frame,
            name,
            (x1 + 18, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2,
        )


def get_toolbar_selection(x, y, frame_width):
    if y > TOOLBAR_HEIGHT:
        return None

    button_w = frame_width // len(TOOLS)
    index = min(x // button_w, len(TOOLS) - 1)
    return TOOLS[index][0]


def save_board(canvas):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(SAVE_FOLDER, f"airteach_board_{timestamp}.png")

    # Save on white background instead of black transparent-looking canvas
    white_bg = np.ones_like(canvas) * 255
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    white_bg[mask > 0] = canvas[mask > 0]

    cv2.imwrite(path, white_bg)
    return path


def overlay_status(output, selected_tool, face_mask_mode):
    """Adds clean demo labels to bottom of screen."""
    cv2.rectangle(output, (0, FRAME_HEIGHT - 55), (FRAME_WIDTH, FRAME_HEIGHT), (20, 20, 20), -1)

    status = f"Tool: {selected_tool} | Face Mask: {face_mask_mode} | m: mask  c: clear  s: save  q: quit"
    cv2.putText(
        output,
        status,
        (20, FRAME_HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )


def main():
    global FACE_MASK_MODE

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Could not open webcam.")
        print("Try changing CAMERA_INDEX from 1 to 0 in app.py.")
        return

    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    selected_tool = "BLUE"
    current_color = COLORS["BLUE"]

    xp, yp = 0, 0
    save_cooldown = 0
    clear_cooldown = 0
    mask_modes = ["PIXELATE", "BLUR", "BLACK"]

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Privacy filter happens before showing frame
            frame = apply_face_mask(frame, FACE_MASK_MODE)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            draw_toolbar(frame, selected_tool)

            if save_cooldown > 0:
                save_cooldown -= 1

            if clear_cooldown > 0:
                clear_cooldown -= 1

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                lm = hand.landmark

                index_tip = (
                    int(lm[8].x * FRAME_WIDTH),
                    int(lm[8].y * FRAME_HEIGHT),
                )
                middle_tip = (
                    int(lm[12].x * FRAME_WIDTH),
                    int(lm[12].y * FRAME_HEIGHT),
                )
                thumb_tip = (
                    int(lm[4].x * FRAME_WIDTH),
                    int(lm[4].y * FRAME_HEIGHT),
                )

                x1, y1 = index_tip
                finger_state = fingers_up(hand)
                thumb_index_dist = distance(index_tip, thumb_tip)

                cv2.circle(frame, index_tip, 12, current_color, -1)

                toolbar_choice = get_toolbar_selection(x1, y1, FRAME_WIDTH)

                # Toolbar selection with index finger near top
                if toolbar_choice and finger_state[1] == 1:
                    selected_tool = toolbar_choice
                    xp, yp = 0, 0

                    if selected_tool in COLORS:
                        current_color = COLORS[selected_tool]

                    elif selected_tool == "ERASER":
                        current_color = (0, 0, 0)

                    elif selected_tool == "CLEAR" and clear_cooldown == 0:
                        canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                        clear_cooldown = 25
                        print("Canvas cleared.")

                    elif selected_tool == "SAVE" and save_cooldown == 0:
                        saved_path = save_board(canvas)
                        print(f"Saved: {saved_path}")
                        save_cooldown = 40

                # Pinch = temporary eraser
                is_pinching = thumb_index_dist < 45

                # Index + middle up = movement mode, no drawing
                movement_mode = finger_state[1] == 1 and finger_state[2] == 1

                # Draw mode = only index up and below toolbar
                draw_mode = finger_state[1] == 1 and finger_state[2] == 0 and y1 > TOOLBAR_HEIGHT

                if movement_mode:
                    xp, yp = 0, 0
                    cv2.rectangle(
                        frame,
                        (x1 - 12, y1 - 12),
                        (middle_tip[0] + 12, middle_tip[1] + 12),
                        (255, 255, 255),
                        2,
                    )

                elif draw_mode:
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    if selected_tool == "ERASER" or is_pinching:
                        cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), ERASER_THICKNESS)
                        cv2.circle(
                            frame,
                            (x1, y1),
                            ERASER_THICKNESS // 2,
                            (255, 255, 255),
                            2,
                        )
                    else:
                        cv2.line(canvas, (xp, yp), (x1, y1), current_color, DRAW_THICKNESS)

                    xp, yp = x1, y1

                else:
                    xp, yp = 0, 0

            # Merge canvas with webcam feed
            gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
            output = cv2.add(frame_bg, canvas_fg)

            draw_toolbar(output, selected_tool)
            overlay_status(output, selected_tool, FACE_MASK_MODE)

            cv2.imshow("AirTeach AI - Virtual Whiteboard", output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("c"):
                canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                print("Canvas cleared.")

            elif key == ord("s"):
                saved_path = save_board(canvas)
                print(f"Saved: {saved_path}")

            elif key == ord("m"):
                current_index = mask_modes.index(FACE_MASK_MODE)
                FACE_MASK_MODE = mask_modes[(current_index + 1) % len(mask_modes)]
                print(f"Face mask mode: {FACE_MASK_MODE}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()