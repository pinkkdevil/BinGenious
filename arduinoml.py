import serial
import time
import numpy as np
from ultralytics import YOLO

# Install serial using pip
# pip install pyserial

# Initialize serial connection (Adjust COM port and baud rate)
ser = serial.Serial('COM5', 9600)
time.sleep(2)

# Load the YOLO model
model_path = r'E:\trained_models\best.pt'  # Path to your trained model
model = YOLO(model_path)

# Function to map class prediction to servo angle
def get_servo_angle(class_id):
    angle_map = {
        0: 0,      # Class 0 -> 0 degrees
        1: 45,     # Class 1 -> 45 degrees
        2: 90,     # Class 2 -> 90 degrees
        3: 135,    # Class 3 -> 135 degrees
        4: 180     # Class 4 -> 180 degrees
    }
    return angle_map.get(class_id, 90)

# Open the webcam for predictions
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break
    
    # Perform model prediction
    results = model(frame)

    for result in results:
        class_id = int(result.boxes.cls.cpu().numpy()[0])
        angle = get_servo_angle(class_id)
        print(f"Class: {class_id}, Angle: {angle}")
        
        # Send angle to Arduino
        ser.write(f'{angle}\n'.encode())
        
    # Display the result
    cv2.imshow('Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()
