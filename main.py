import mss
import torch
import torch_directml
import pygetwindow as gw
import numpy as np
import cv2
import time
from ultralytics import YOLO

from AI_Models.VAE.models import VAE
#from AI_Models.EntityDetector import 

device = torch_directml.device()
entity_detector_resolution = (480, 224)
vae_resolution = (144, 64)
topbar_size = 32
fps = 10
interval = 1 / fps

entity_detector_args = {
    "path": "./AI_Models/EntityDetector/model.pt",
    "threshold": 0.5,
    "iou_threshold": 0.4,
    "labels": [
        "Ball",
        "Enemy",
        "Friendly",
        "Gem",
        "Hot_Zone",
        "Me",
        "PP",
        "PP_Box",
        "Safe_Enemy",
        "Safe_Friendly"
    ],
    "label_colors": {
        "Ball": (255, 0, 0),
        "Enemy": (0, 0, 255),
        "Friendly": (0, 255, 0),
        "Gem": (0, 255, 255),
        "Hot_Zone": (255, 165, 0),
        "Me": (255, 255, 255),
        "PP": (255, 0, 255),
        "PP_Box": (255, 255, 0),
        "Safe_Enemy": (128, 0, 128),
        "Safe_Friendly": (0, 128, 0),
    }
}

vae_args = {
    "device": device,
    "disentangle": True,
    "resolution": [vae_resolution[1], vae_resolution[0]],
    "latent_size": 2048,
    "path": "./AI_Models/VAE/model.pt"
}




vae_model = VAE(vae_args, 0).to(device)
checkpoint = torch.load(vae_args["path"])
vae_model.load_state_dict(checkpoint['model_state_dict'])
vae_model.eval()
vae_model.decoder.eval()
vae_model.encoder.eval()
for param in vae_model.parameters():
    param.requires_grad = False

entity_model = YOLO(entity_detector_args["path"], verbose=False)

def get_vae_output(frame):
    vae_img = cv2.resize(frame, vae_resolution, interpolation=cv2.INTER_LANCZOS4)
    vae_img = torch.tensor(vae_img, dtype=torch.float32).permute(2, 0, 1)
    vae_img = vae_img / 127.5 - 1
    vae_img = vae_img.unsqueeze(0).to(device)

    mu = vae_model.encoder.forward_mu(vae_img)
    return mu

def apply_nms(boxes, scores, labels):
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=entity_detector_args["threshold"], nms_threshold=entity_detector_args["iou_threshold"])

    nms_detections = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            score = scores[i]
            label = labels[i]

            nms_detections.append({
                "label": label,
                "confidence": score,
                "bbox": [x, y, w, h]
            })

    return nms_detections

def get_entities(frame):
    """frame_resize = cv2.resize(frame, entity_detector_resolution, interpolation=cv2.INTER_LANCZOS4)
    frame_resize = np.transpose(frame_resize, (2, 0, 1))
    frame_resize = frame_resize / 255.0

    frame_resize = np.asarray(frame_resize, dtype=np.float32)
    frame_resize = np.expand_dims(frame_resize, axis=0)"""

    frame_resize = cv2.resize(frame, entity_detector_resolution, interpolation=cv2.INTER_LANCZOS4)
    frame_resize = torch.tensor(frame_resize, dtype=torch.float32).permute(2, 0, 1)
    frame_resize = frame_resize / 255.0
    frame_resize = frame_resize.unsqueeze(0)#.to(device)

    start = time.time()
    #preds = entity_model.predict(frame_resize, verbose=False, conf=entity_detector_args["threshold"], iou=entity_detector_args["iou_threshold"], imgsz=[entity_detector_resolution[1], entity_detector_resolution[0]])[0]
    preds = entity_model.predict(frame_resize, verbose=False, conf=entity_detector_args["threshold"], iou=entity_detector_args["iou_threshold"], device="cuda:0")[0]
    outputs = []

    #for result in preds:
    for detection in preds.boxes:
        outputs.append({
            "label": entity_detector_args["labels"][int(detection.cls)],
            "confidence": float(detection.conf),
            "bbox": detection.xyxyn
        })

    print((time.time() - start) * 1000)

    return outputs

    """bboxes_out = []
    scores_out = []
    labels_out = []

    for idx in range(preds.shape[2]):
        data = preds[0, :, idx]

        box = data[:4]
        scores = data[4:]

        score = np.max(scores) 
        label = np.argmax(scores)

        if score > entity_detector_args["threshold"]: 
            x_center, y_center, width, height = box
            x = x_center - 0.5 * width
            y = y_center - 0.5 * height
            
            bboxes_out.append([float(x/frame_resize.shape[3]), float(y/frame_resize.shape[2]), float(width/frame_resize.shape[3]), float(height/frame_resize.shape[2])])
            labels_out.append(entity_detector_args["labels"][int(label)])
            scores_out.append(float(score))

    return apply_nms(bboxes_out, scores_out, labels_out)"""
"""
    decoded_image = vae_model.decoder(vae_data)
    decoded_image = decoded_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    decoded_image = (decoded_image + 1) * 127.5 
    decoded_image = np.clip(decoded_image, 0, 255).astype(np.uint8)
    decoded_image = cv2.resize(decoded_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGRA)
"""

def handle_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR).astype(np.uint8)

    entities = get_entities(frame_rgb)
    vae_data = get_vae_output(frame_rgb)

    for entity in entities:
        label = entity['label']
        confidence = entity['confidence']
        bbox = entity['bbox'][0]

        x1 = int(bbox[0] * frame.shape[1])
        y1 = int(bbox[1] * frame.shape[0])
        x2 = int(bbox[2] * frame.shape[1])
        y2 = int(bbox[3] * frame.shape[0])

        color = entity_detector_args["label_colors"][label]

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 1)

        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow(f"Capture - {app_name}", frame_bgr)

def capture_app_window(app_name):
    windows = gw.getWindowsWithTitle(app_name)
    if not windows:
        print(f"Application with title '{app_name}' not found.")
        return

    window = windows[0]
    bbox = {
        "left": window.left,
        "top": window.top,
        "width": window.width,
        "height": window.height
    }

    with mss.mss() as sct:
        last_fire_time = time.time()

        while True:
            frame = sct.grab(bbox)

            img = np.array(frame)[:, :, :3]
            img = img[topbar_size:, :, :]

            if time.time() - last_fire_time >= interval:
                last_fire_time = time.time()
                handle_frame(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    app_name = "BlueStacks App Player"
    capture_app_window(app_name)
