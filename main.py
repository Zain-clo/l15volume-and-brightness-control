import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc


# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# Get computer volume control
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume.QueryInterface(IAudioEndpointVolume)

min_volume, max_volume = volume.GetVolumeRange()[:2]


# Start camera
cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip camera
    frame = cv2.flip(frame, 1)

    # Get frame size
    height, width = frame.shape[:2]

    # Detect hands
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)


    if results.multi_hand_landmarks:

        for hand, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            # Draw hand
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Get left/right hand
            label = hand_info.classification[0].label

            # Thumb and index finger
            thumb = hand.landmark[4]
            index = hand.landmark[8]

            # Convert coordinates to pixels
            thumb_x = int(thumb.x * width)
            thumb_y = int(thumb.y * height)

            index_x = int(index.x * width)
            index_y = int(index.y * height)

            # Draw points and line
            cv2.circle(
                frame,
                (thumb_x, thumb_y),
                10,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                (index_x, index_y),
                10,
                (255, 0, 0),
                -1
            )

            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (0, 255, 0),
                3
            )

            # Calculate distance
            distance = np.hypot(
                index_x - thumb_x,
                index_y - thumb_y
            )


            # LEFT hand = volume
            if label == "Left":

                volume_level = np.interp(
                    distance,
                    [30, 300],
                    [min_volume, max_volume]
                )

                volume.SetMasterVolumeLevel(
                    volume_level,
                    None
                )

                volume_percent = int(
                    np.interp(distance, [30, 300], [0, 100])
                )

                cv2.putText(
                    frame,
                    f"Volume: {volume_percent}%",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2
                )


            # RIGHT hand = brightness
            elif label == "Right":

                brightness = int(
                    np.interp(
                        distance,
                        [30, 300],
                        [0, 100]
                    )
                )

                sbc.set_brightness(brightness)

                cv2.putText(
                    frame,
                    f"Brightness: {brightness}%",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )


    # Show camera
    cv2.imshow("Hand Gesture Control", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()