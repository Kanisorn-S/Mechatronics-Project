import mediapipe as mp
import cv2 as cv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def detect_index_finger(frame):
  hands = mp_hands.Hands()

  frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
  results = hands.process(frame)
  frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
  frame_height, frame_width, _ = frame.shape
  frame_with_finger = frame.copy()
  if results.multi_hand_world_landmarks:
    hand_landmark = results.multi_hand_landmarks[0]
    x = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width
    y = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height
    mp_drawing.draw_landmarks(
      frame_with_finger,
      hand_landmark,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style() 
    )
    return frame_with_finger, (x, y)
  else:
    return frame_with_finger, (0, 0)

    
if __name__ == "__main__":
  cap = cv.VideoCapture(0)
  hands = mp_hands.Hands()

  while cap.isOpened():
    ret, frame = cap.read()
    frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape
    if results.multi_hand_world_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        print("X: ", (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width))
        print("Y: ", (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height))
        mp_drawing.draw_landmarks(
          frame,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS
        )
    cv.imshow("Handtracker", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
    
  cap.release()
  cv.destroyAllWindows()
    