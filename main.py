import mss
import torch
import platform
import pywinctl as pwc
import numpy as np
import cv2
import time
#import subprocess
import uuid
import easyocr
import matplotlib.pyplot as plt
#import adbutils
from ultralytics import YOLO

from AI_Models.VAE.models import VAE
#from AI_Models.EntityDetector import 

def get_device():
    if torch.cuda.is_available():
        is_nvidia = False

        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            properties = torch.cuda.get_device_properties(i)
            if "NVIDIA" in properties.name.upper():
                is_nvidia = True

        if(is_nvidia):
            print("Using NVIDIA GPU with CUDA.")
        else:
            print("Using AMD GPU with ROCm.")

        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple Metal Performance Shaders (MPS).")
        return torch.device("mps")
    elif platform.system() == "Windows":
        import torch_directml
        print("Using UNKNOWN GPU with DirectML")
        return torch_directml.device()
    else:
        print("No GPU detected. Using CPU.")
        return torch.device("cpu")

device = get_device()
ocr = easyocr.Reader(['en'])#, detector=False)
app_name = "BlueStacks"
#ocr_detector_resolution = (1440, 672)
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
checkpoint = torch.load(vae_args["path"])#, map_location=device)
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
    frame_resize = cv2.resize(frame, entity_detector_resolution, interpolation=cv2.INTER_LANCZOS4)
    frame_resize = torch.tensor(frame_resize, dtype=torch.float32).permute(2, 0, 1)
    frame_resize = frame_resize / 255.0
    frame_resize = frame_resize.unsqueeze(0)#.to(device)

    #start = time.time()
    #preds = entity_model.predict(frame_resize, verbose=False, conf=entity_detector_args["threshold"], iou=entity_detector_args["iou_threshold"], imgsz=[entity_detector_resolution[1], entity_detector_resolution[0]])[0]
    preds = entity_model.predict(frame_resize, verbose=False, conf=entity_detector_args["threshold"], iou=entity_detector_args["iou_threshold"])[0]
    outputs = []

    #for result in preds:
    for detection in preds.boxes:
        outputs.append({
            "label": entity_detector_args["labels"][int(detection.cls)],
            "confidence": float(detection.conf),
            "bbox": detection.xyxyn
        })

    #print((time.time() - start) * 1000)

    return outputs

def transform_string_to_number(s):
    numeric_part = ''.join(char for char in s if char.isdigit())
    return int(numeric_part) if numeric_part else 0

def handle_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR).astype(np.uint8)
    #frame_ocr = cv2.resize(frame_rgb, ocr_detector_resolution, interpolation=cv2.INTER_LANCZOS4)

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

        #x1_ocr = int(bbox[0] * frame_ocr.shape[1])
        #y1_ocr = int(bbox[1] * frame_ocr.shape[0])
        #x2_ocr = int(bbox[2] * frame_ocr.shape[1])
        #y2_ocr = int(bbox[3] * frame_ocr.shape[0])

        color = entity_detector_args["label_colors"][label]


        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        text = f"{label}: {confidence:.2f}"
        cv2.putText(frame_bgr, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if label in ["Enemy", "Friendly", "Me"]:
            #roi = frame_ocr[y1_ocr:y2_ocr, x1_ocr:x2_ocr]
            roi = frame_rgb[y1-3:y2+4, x1-3:x2+4]
            health = None

            """white_mask = (
                (roi[:, :, 0] > 200) & 
                (roi[:, :, 1] > 200) & 
                (roi[:, :, 2] > 200) &
                (np.abs(roi[:, :, 0] - roi[:, :, 1]) < 10) &
                (np.abs(roi[:, :, 0] - roi[:, :, 2]) < 10) &
                (np.abs(roi[:, :, 1] - roi[:, :, 2]) < 10)
            )

            health_ocr_roi = np.zeros_like(roi)
            health_ocr_roi[white_mask] = roi[white_mask]
            white_coords = np.where(health_ocr_roi)
            if len(white_coords[0]) > 0 and len(white_coords[1]) > 0:
                min_y, max_y = np.min(white_coords[0]), np.max(white_coords[0])
                min_x, max_x = np.min(white_coords[1]), np.max(white_coords[1])
                cropped_health_ocr_roi = health_ocr_roi[min_y-3:max_y+4, min_x-3:max_x+4]
                if cropped_health_ocr_roi.shape[0] > 0 and cropped_health_ocr_roi.shape[1] > 0:
                    #start = time.time()
                    
                    ocr_results = ocr.readtext(
                        cropped_health_ocr_roi, 
                        allowlist="0123456789",
                        #paragraph=True,
                        #detail=False,
                        #decoder="beamsearch",
                    )

                    text = len(ocr_results) > 0 and ocr_results[0] or None
                    health = int(text) if text else 0
                    
                    #roi_height = cropped_health_ocr_roi.shape[0]
                    #y_offset = y1 + 30
                    #x_offset = x2 + 5
                    #frame_bgr[y_offset: y_offset + roi_height, x_offset: x_offset + cropped_health_ocr_roi.shape[1]] = cropped_health_ocr_roi"""

            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue

            ocr_results = ocr.readtext(
                roi, 
                #allowlist="0123456789",
                #paragraph=True,
                #detail=False,
                #decoder="beamsearch",
            )

            for ocr_result in ocr_results:
                bounding_box = ocr_result[0]
                text = ocr_result[1]
                confidence = ocr_result[2]

                text_as_health = transform_string_to_number(text)
                if text_as_health > 0:
                    x1_ocr = int(bounding_box[0][0])
                    y1_ocr = int(bounding_box[0][1])
                    x2_ocr = int(bounding_box[2][0])
                    y2_ocr = int(bounding_box[2][1])

                    ocr_roi = roi[y1_ocr:y2_ocr+1, x1_ocr:x2_ocr+1]

                    white_mask = (
                        (ocr_roi[:, :, 0] > 150) & 
                        (ocr_roi[:, :, 1] > 150) & 
                        (ocr_roi[:, :, 2] > 150)
                    )

                    white_pixel_count = np.sum(white_mask)
                    total_pixel_count = ocr_roi.shape[0] * ocr_roi.shape[1]
                    white_percentage = white_pixel_count / total_pixel_count

                    if white_percentage > 0.2:
                        health = text_as_health

                    #cv2.rectangle(frame_bgr, (x1_ocr + x1, y1_ocr + y1), (x2_ocr + x1, y2_ocr + y1), (255, 0, 0), 2)

            if health is not None:
                health_text = f"Health: {health}"
                cv2.putText(frame_bgr, health_text, (x2 + 5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow(f"Capture - {app_name}", cv2.resize(frame_bgr, [480 * 4, 224 * 4], interpolation=cv2.INTER_LANCZOS4))

def find_window(app_name):
    windows = pwc.getAllWindows()

    for win in windows:
        if app_name.lower() in win.title.lower():
            return win
    return None


def capture_app_window(app_name):
    """
    adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
    device = adb.device()"""

    window = find_window(app_name)
    if not window:
        print(f"Unable to find {app_name}...")
        return
    
    bbox = window.box
    if bbox.left < 0 or bbox.top < 0:
        print("Invalid Bbox, please focus again on Brawl.")
        return

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
    capture_app_window(app_name)
