## conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
## pip install ultralytics


import cv2
import numpy as np
from ultralytics import RTDETR
import time

# Load a model
model = RTDETR('rtdetr-l.pt')  # load an official model
# model = RTDETR('rtdetr-x.pt')  # load an official model

# Display model information (optional)
model.info()
model.overrides['conf'] = 0.3  # NMS confidence threshold
model.overrides['iou'] = 0.4  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image
model.overrides['classes'] = [2,3,0] ## define classes


coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def process(image, track=True):
    original_size = image.shape[:2]

    # Predict with the model
    results = model.track(image, verbose=False, device=0, persist=True, tracker="bytetrack.yaml") if track else model.predict(image, verbose=False, device=0)

    # Iterate over detection results
    for detection in results:
        boxes = detection.boxes.xyxy
        names = detection.names
        classes = detection.boxes.cls
        confs = detection.boxes.conf
        ids = detection.boxes.id if track else [[None]*len(boxes)]

        if ids is None:
            continue

        # Iterate over rows of the tensor
        for box, name, class_idx, conf, id_ in zip(boxes, names, classes, confs, ids):
            # Convert box coordinates to integers
            xmin, ymin, xmax, ymax = map(int, box)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 30, 255), 2)

            class_name = coco_classes[int(class_idx)]
            label = f' ID: {int(id_)} ' if track and id_ is not None else ''
            label += f'{class_name}: {round(float(conf) * 100, 1)}%'

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
            dim, baseline = text_size[0], text_size[1]
            cv2.rectangle(image, (int(xmin), int(ymin)), ((int(xmin) + dim[0] // 3) - 20, int(ymin) - dim[1] + baseline), (30, 30, 30), cv2.FILLED)
            cv2.putText(image, label, (int(xmin), int(ymin) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image




cap = cv2.VideoCapture('test.mp4')


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec if needed (e.g., 'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, 15, (frame_width, frame_height))

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frameId = 0
start_time = time.time()
fps = str()

while True:
    frameId += 1
    ret, frame = cap.read()
    if not ret:
        break

    frame = process(frame, track=True)

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if frameId % 10 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps_current = 10 / elapsed_time  # Calculate FPS over the last 20 frames
        fps = f'FPS: {fps_current:.2f}'
        start_time = time.time()  # Reset start_time for the next 20 frames

    cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("rt-detr", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture featuresect
cap.release()
out.release()
cv2.destroyAllWindows()
