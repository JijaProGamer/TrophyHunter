import threading
import torch
import platform
import numpy as np
import cv2
import time
import adbutils
import subprocess
from adbutils import adb
from ultralytics import YOLO

from AI_Models.VAE.models import VAE

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
        print("No NVIDIA GPU detected. Using DirectML.")
        return torch_directml.device()
    else:
        print("No GPU detected. Using CPU.")
        return torch.device("cpu")

device = get_device()
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
checkpoint = torch.load(vae_args["path"], map_location=device)
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


def get_entities(frame):
    frame_resize = cv2.resize(frame, entity_detector_resolution, interpolation=cv2.INTER_LANCZOS4)
    frame_resize = torch.tensor(frame_resize, dtype=torch.float32).permute(2, 0, 1)
    frame_resize = frame_resize / 255.0
    frame_resize = frame_resize.unsqueeze(0)#.to(device)

    start = time.time()
    #preds = entity_model.predict(frame_resize, verbose=False, conf=entity_detector_args["threshold"], iou=entity_detector_args["iou_threshold"], imgsz=[entity_detector_resolution[1], entity_detector_resolution[0]])[0]
    preds = entity_model.predict(frame_resize, verbose=False, conf=entity_detector_args["threshold"], iou=entity_detector_args["iou_threshold"], device="cuda:0")[0]
    outputs = []

    for detection in preds.boxes:
        outputs.append({
            "label": entity_detector_args["labels"][int(detection.cls)],
            "confidence": float(detection.conf),
            "bbox": detection.xyxyn
        })

    #print((time.time() - start) * 1000)

    return outputs

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

    cv2.imshow(f"TrophyHunter", frame_bgr)


def capture_app_window():
    adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
    #device = adb.device(serial=adb.list()[0]) # only first device for now 
    device = adb.device()

    local_scrcpy_path = "./scrcpy-server.jar"
    remote_scrcpy_path = "/data/local/tmp/vlc-scrcpy-server.jar"

    device.sync.push(local_scrcpy_path, remote_scrcpy_path)
    device.forward("tcp:1234", "localabstract:scrcpy")

    def run_scrcpy_server():
        output = device.shell(
            "CLASSPATH={remote_scrcpy_path} \     app_process / com.genymobile.scrcpy.Server 1.25 \     tunnel_forward=true control=false cleanup=false \     max_size=720 raw_video_stream=true"
        )

    run_scrcpy_server()
    #scrcpy_thread = threading.Thread(target=run_scrcpy_server, daemon=True)
    #scrcpy_thread.start()

    """"
    ffmpeg_process = subprocess.Popen(
        ["ffmpeg", 
         "-i", "pipe:0",
         "-pix_fmt", "rgb24",
         "-vcodec", "rawvideo", 
         
         "-an", "-sn","video.mkv"# "-f", "rawvideo", "-"
        ],
        stdin=output.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )"""
 
            #if time.time() - last_fire_time >= interval:
            #    last_fire_time = time.time()
            #    handle_frame(img)

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

        #cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_app_window()
