# %%
from random import randint

import cv2
import numpy as np

red = (255, 0, 0)
yellow = (255, 255, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

font = cv2.FONT_HERSHEY_PLAIN

prototext = "MobileNetSSD_deploy.prototxt"
weights = "MobileNetSSD_deploy.caffemodel"

# %%
def calculate_intersection_ratio(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2 = x1_1 + w1, y1_1 + h1

    x2_1, y2_1, w2, h2 = box2
    x2_2, y2_2 = x2_1 + w2, y2_1 + h2

    x_intersection1 = max(x1_1, x2_1)
    y_intersection1 = max(y1_1, y2_1)
    x_intersection2 = min(x1_2, x2_2)
    y_intersection2 = min(y1_2, y2_2)

    intersection_width = max(0, x_intersection2 - x_intersection1)
    intersection_height = max(0, y_intersection2 - y_intersection1)

    intersection_area = intersection_width * intersection_height

    area_box1 = w1 * h1
    area_box2 = w2 * h2

    intersection_ratio_box1 = intersection_area / area_box1
    intersection_ratio_box2 = intersection_area / area_box2

    return intersection_ratio_box1, intersection_ratio_box2

# %%
video_path = 'test_video1.mp4'
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

thr = 0.3
psr = 0.05

person_class_id = 15


# %%
def detect_pedestrians(frame):
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300),
                                 (127.5, 127.5, 127.5), False)
    boxes = []
    net.setInput(blob)
    detections = net.forward()

    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > thr:
            class_id = int(detections[0, 0, i, 1])

            x1 = int(detections[0, 0, i, 3] * cols)
            y1 = int(detections[0, 0, i, 4] * rows)
            x2 = int(detections[0, 0, i, 5] * cols)
            y2 = int(detections[0, 0, i, 6] * rows)

            h_scale = frame.shape[0] / 300.0
            w_scale = frame.shape[1] / 300.0

            x1 = int(w_scale * x1)
            y1 = int(h_scale * y1)
            x2 = int(w_scale * x2)
            y2 = int(h_scale * y2)

            if class_id == person_class_id:
                boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


# %%
tracker_dict = {}
tracker_abort = {}
tracker_suc_count = {}
user_position = {}
tracker_rects = {}
tracker_state = {}

out = cv2.VideoWriter('out_video_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

ret, first_frame = cap.read()

colors = []

overlap_ratio = 0.1

net = cv2.dnn.readNetFromCaffe(prototext, weights)

# %%

first_frame_boxes = detect_pedestrians(first_frame)

skip_frames = 10
skip_count = 0
people_id = 0
for bbox in first_frame_boxes:
    tracker_params = cv2.TrackerCSRT.Params()
    tracker_params.psr_threshold = psr
    tracker = cv2.TrackerCSRT.create(tracker_params)
    tracker.init(first_frame, bbox)

    tracker_abort[people_id] = False
    tracker_suc_count[people_id] = 0
    tracker_dict[people_id] = tracker
    tracker_rects[people_id] = ()
    tracker_state[people_id] = False
    user_position[people_id] = []

    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    people_id += 1

while cap.isOpened():
    ret, frame = cap.read()
    skip_count += 1
    if not ret:
        break

    # update tracker
    for i, tracker in tracker_dict.items():
        if tracker_abort[i]:
            continue
        spec_color = colors[i]
        ret, newbox = tracker.update(frame)
        tracker_state[i] = ret
        tracker_rects[i] = newbox
        if not ret:
            continue
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3])

        cv2.rectangle(frame, p1, p2, spec_color, 2, 1)
        cv2.putText(frame, f'ID of people {i + 1}', p1, font, 0.9, green)

        # record path of movement
        x, y, w, h = newbox
        center_x = x + w / 2
        center_y = y + h / 2
        bottom_center = (center_x, y + h)
        user_position[i].append(bottom_center)

        # draw path
        path = user_position[i]
        smoothed_path = []
        if len(path) > 1:
            window_size = 15
            for i in range(len(path)):
                start = max(0, i - window_size // 2)
                end = min(len(path), i + window_size // 2 + 1)
                smoothed_point = np.mean(path[start:end], axis=0)
                smoothed_path.append(smoothed_point)

        if len(smoothed_path) > 1:
            cv2.polylines(frame, [np.array(smoothed_path, dtype=np.int32)],
                          isClosed=False,
                          color=spec_color, thickness=5)

    potential_new = detect_pedestrians(frame)

    if skip_count > skip_frames:
        skip_count = 0
        for i, tracker in tracker_dict.items():
            if tracker_suc_count[i] < skip_count // 3:
                tracker_abort[i] = True
            tracker_suc_count[i] = 0

        for new_box in potential_new:
            is_new = True

            for id, exist_box in tracker_rects.items():
                if (not tracker_state[id]) or (tracker_abort[id]):
                    continue
                r1, r2 = calculate_intersection_ratio(new_box, exist_box)
                if r1 > overlap_ratio or r2 > overlap_ratio:
                    tracker_suc_count[id] += 1
                    tracker_dict[id].init(frame, new_box)
                    is_new = False
                    break

            if is_new:
                tracker_params = cv2.TrackerCSRT.Params()
                tracker_params.psr_threshold = psr
                tracker = cv2.TrackerCSRT.create(tracker_params)
                tracker.init(frame, new_box)

                tracker_abort[people_id] = False
                tracker_suc_count[people_id] = 0
                tracker_dict[people_id] = tracker
                tracker_rects[people_id] = ()
                tracker_state[people_id] = False
                user_position[people_id] = []
                colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
                people_id += 1

    cv2.imshow('MultiTracker', frame)

    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# %%
