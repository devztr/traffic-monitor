import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
video_path = "traffic.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    exit()

H, W = frame.shape[:2]
LINE_Y = int(H // 2)

cap.release()

# --- Labels and state ---
labels = {2: "Car", 5: "Bus", 7: "Truck"}

track_positions = {}
crossed_ids     = {}
in_counts       = {2: 0, 5: 0, 7: 0}
out_counts      = {2: 0, 5: 0, 7: 0}


def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def remove_duplicate_boxes(boxes, track_ids, classes, iou_threshold=0.4):
    keep, suppressed = [], set()
    for i in range(len(boxes)):
        if i in suppressed:
            continue
        keep.append(i)
        for j in range(i + 1, len(boxes)):
            if j not in suppressed and compute_iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed.add(j)
    return boxes[keep], track_ids[keep], classes[keep]


# --- Main loop ---
for result in model.track(source=video_path, stream=True, classes=[2, 5, 7],
                           persist=True, imgsz=640):
    frame = result.orig_img.copy()
    cv2.line(frame, (0, LINE_Y), (W, LINE_Y), (0, 0, 255), 2)

    if result.boxes is None or result.boxes.id is None:
        cv2.imshow("Traffic Monitor", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    boxes     = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    classes   = result.boxes.cls.cpu().numpy().astype(int)
    boxes, track_ids, classes = remove_duplicate_boxes(boxes, track_ids, classes)

    for i, (box, tid) in enumerate(zip(boxes, track_ids)):
        cx, cy = get_center(box)
        x1, y1, x2, y2 = map(int, box)
        cls = classes[i]

        if tid in track_positions:
            prev_y = track_positions[tid]
            if prev_y < LINE_Y <= cy and tid not in crossed_ids:
                in_counts[cls] += 1
                crossed_ids[tid] = "in"
            elif prev_y > LINE_Y >= cy and tid not in crossed_ids:
                out_counts[cls] += 1
                crossed_ids[tid] = "out"

        track_positions[tid] = cy

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 165, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    # Dashboard
    cv2.rectangle(frame, (10, 10), (340, 30 + len(labels) * 30), (0, 0, 0), -1)
    y = 35
    for cls_id, name in labels.items():
        cv2.putText(frame, f"{name:<12} In: {in_counts[cls_id]}  Out: {out_counts[cls_id]}",
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 30

    cv2.imshow("Traffic Monitor", frame)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

# --- Summary Chart ---
names      = [labels[c] for c in labels]
ins        = [in_counts[c] for c in labels]
outs       = [out_counts[c] for c in labels]
totals     = [ins[i] + outs[i] for i in range(len(names))]

x = np.arange(len(names))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width, ins,    width, label="IN",    color="#4CAF50")
ax.bar(x,         outs,   width, label="OUT",   color="#F44336")
ax.bar(x + width, totals, width, label="TOTAL", color="#2196F3")

ax.set_title("Traffic Summary", fontsize=14, fontweight="bold")
ax.set_xlabel("Vehicle Type")
ax.set_ylabel("Count")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.grid(axis="y", alpha=0.4)

for i, (a, b, t) in enumerate(zip(ins, outs, totals)):
    ax.text(i - width, a + 0.2, str(a), ha="center", fontsize=9)
    ax.text(i,         b + 0.2, str(b), ha="center", fontsize=9)
    ax.text(i + width, t + 0.2, str(t), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("traffic_summary.png", dpi=150)
plt.show()
print("Summary saved to traffic_summary.png")