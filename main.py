import cv2
import mediapipe as mp
import math
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities , IAudioEndpointVolume
from comtypes import CLSCTX_ALL

devices = AudioUtilities.GetSpeakers()

volume = devices.EndpointVolume

mp_hands= mp.solutions.hands
mp_draw=mp.solutions.drawing_utils

hands=mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence =0.7,
    min_tracking_confidence= 0.7

)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: 
        break
    frame = cv2.flip(frame,1)
    rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results= hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            mp_draw.draw_landmarks(
                frame, hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

            label = handedness.classification[0].label

            thumb = hand_landmarks.landmark[4]
            index= hand_landmarks.landmark[8]

            dx = thumb.x -index.x
            dy = thumb.y - index.y

            distance = math.hypot(dx,dy)

            if label == "Right":
                brightness = int(
                     max(0, (min(100, distance *300)))
                )
                sbc.set_brightness(brightness)
                cv2.putText(
                    frame,
                    f"Brightness:{brightness}%",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )

            elif label == "Left":
                   vol = max(-65, min (0,(distance*100) - 65))
                   volume.setMasterVolumeLevel(vol, None)

                   volume_perc= int(
                        ((volume + 65)/65) *100
                   )
                   cv2.putText(
                        frame,
                        f"Volume:{volume_perc}%",
                        (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2
                   )

        cv2.imshow("Hand Gesture control", frame)

        if cv2.waitkey(1) & 0xFF == ord("q"):
             break

cap.release()
cv2.destroyAllWindows()