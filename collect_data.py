import cv2
import os

gesture = input("Enter gesture name: ")

save_path = os.path.join("dataset", gesture)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found!")
    exit()

count = len(os.listdir(save_path))

print("Press 'S' to save an image")
print("Press 'Q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        f"{gesture} : {count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()