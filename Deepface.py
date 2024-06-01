from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()  
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    try:
        res = DeepFace.analyze(frame_rgb, actions=['emotion'])
        print(res)

        emotion = res['dominant_emotion']
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error: {e}")

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
