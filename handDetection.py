import cv2
import mediapipe
from pynput.mouse import Button, Controller

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

capture = cv2.VideoCapture(1)

frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

mouse = Controller()

def isClick(hand_landmarks_landmark):
    normalizeMidTip  = hand_landmarks.landmark[handsModule.HandLandmark.MIDDLE_FINGER_TIP]
    normalizeMidPalm = hand_landmarks.landmark[handsModule.HandLandmark.MIDDLE_FINGER_MCP]
    if normalizeMidTip.y > normalizeMidPalm.y:
        return True
    return False

isPressed = False
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while(True):
        if cv2.waitKey(1) == 27:
            break
        ret, frame = capture.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None:
            for hand_landmarks in results.multi_hand_landmarks:
                normalizedLandmark = hand_landmarks.landmark[handsModule.HandLandmark.INDEX_FINGER_TIP]
                coordinate = drawingModule._normalized_to_pixel_coordinates(\
                            normalizedLandmark.x,
                            normalizedLandmark.y,
                            frameWidth,
                            frameHeight)
                moveScale = 1
                center = (int(frameWidth/2), int(frameHeight/2))
                flipped = (frameWidth - coordinate[0], coordinate[1]) #horizontal flip
                vector = (moveScale*(flipped[0] - center[0]), moveScale*(flipped[1] - center[1]))
                show = (center[0] + vector[0], center[1] + vector[1])
                cv2.circle(frame, coordinate, 5, (0, 255, 0), -1)
                if show[0]>0 and show[0]<frameWidth and show[1]>0 and show[1]<frameWidth:
                    mouse.position = show
                if isClick(hand_landmarks.landmark) is True:
                    isPressed = True
                    mouse.press(Button.left)
                if isPressed is True:
                    isPressed = False
                    mouse.release(Button.left)
                #drawingModule.draw_landmarks(frame, hand_landmarks, handsModule.HAND_CONNECTIONS)
        #frame = cv2.flip(frame, 1)
        #cv2.imshow("Hand", frame)
#cv2.destroyAllWindows()
capture.release()
