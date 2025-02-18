import cv2
import numpy as np

def get_perspective_transform(image, contour):
    rect = cv2.minAreaRect(contour)
    box = np.intp(cv2.boxPoints(rect))

    box = sorted(box, key=lambda x: (x[1], x[0]))
    if box[0][0] > box[1][0]:
        box[0], box[1] = box[1], box[0]
    if box[2][0] > box[3][0]:
        box[2], box[3] = box[3], box[2]

    # The size of the transformed rectangle
    width = 200
    height = 200
    dst_points = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(np.array(box, dtype="float32"), dst_points)

    # perspective transformation
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped

def detect_arrow_direction(contour):
    hull = cv2.convexHull(contour)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    distances = []
    for i in range(len(approx)):
        for j in range(i+1, len(approx)):
            dist = np.linalg.norm(approx[i] - approx[j])
            distances.append((dist, approx[i][0], approx[j][0]))

    distances.sort(reverse=True, key=lambda x: x[0])
    if not distances:
        return "No Arrow Detected"

    pt1, pt2 = distances[0][1], distances[0][2]

    vector = np.array(pt2) - np.array(pt1)
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi

    print(f"angle: {angle}")

    if angle < -180:
        angle += 360
    elif angle > 180:
        angle -= 360

    #  direction decision
    if 30 <= angle <= 60:
        return "Forward"
    elif 120 <= angle <= 160:
        return "Right"
    elif -145 <= angle <= -120 or 145 <= angle <= 170:
        return "Left"

    return "Unknown"

def process_frame(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(binary, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if not contours:
        return frame, "No Arrow Detected"

    valid_contours = [c for c in contours if 5000 < cv2.contourArea(c) < 100000]
    if not valid_contours:
        return frame, "No Arrow Detected"

    largest_contour = max(valid_contours, key=cv2.contourArea)

    processed_frame = get_perspective_transform(frame, largest_contour)
    arrow_direction = detect_arrow_direction(largest_contour)

    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
    cv2.putText(frame, f"Direction: {arrow_direction}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame, arrow_direction

def main():
    cap = cv2.VideoCapture(0)


    while True:
        ret, frame = cap.read()

        processed_frame, arrow_direction = process_frame(frame)

        cv2.imshow("Arrow Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()