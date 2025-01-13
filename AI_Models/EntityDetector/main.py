from ultralytics import YOLO
import numpy as np
import easyocr
import os 
import yaml
import cv2

health_values = {
    "Me": {
        "lower_bound": np.array([90 * 255/360, 55 * 255/100, 70 * 255/100]),
        "upper_bound": np.array([140 * 255/360, 100 * 255/100, 100 * 255/100]),
    },
    "Enemy": {
        "lower_bound": np.array([310 * 255/360, 70 * 255/100, 70 * 255/100]),
        "upper_bound": np.array([370 * 255/360, 100 * 255/100, 100 * 255/100]),
    },
    "Friendly": {
        "lower_bound": np.array([170 * 255/360, 80 * 255/100, 80 * 255/100]),
        "upper_bound": np.array([210 * 255/360, 100 * 255/100, 100 * 255/100]),
    }
}

def transform_string_to_number(s):
    replacements = {
        'a': '4',
        'A': '4',
        'z': '2',
        'o': '0',
        'O': '0',
        'l': '1',
        'L': '1',
        'I': '1',
        'b': '8',
        'g': '9',
        't': '1',
        'T': '1',
        'j': '1',
        'J': '1',
        'n': '1',
        's': '5',
        'S': '5'
    }
    
    for key, value in replacements.items():
        s = s.replace(key, value)
    
    numeric_part = ''.join(char for char in s if char.isdigit())
    return int(numeric_part) if numeric_part else 0

class EntityDetector():
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(dir_path, 'hyperparameters.yaml')) as f:
            args = yaml.safe_load(f)

        self.args = args
        self.entity_model = YOLO(os.path.join(dir_path, args["path"]), verbose=False)
        self.ocr = easyocr.Reader(['en'])
    def predict(self, frame_bgr, frame_rgb):
        raw_predictions = self.raw_predict(frame_bgr, frame_rgb)
        predictions = self.augment_predictions(frame_bgr, frame_rgb, raw_predictions)

        return predictions
    def raw_predict(self, frame_bgr, frame_rgb):
        #preds = entity_model.track(frame_resize, verbose=False, conf=entity_detector_args["threshold"], iou=entity_detector_args["iou_threshold"], tracker="botsort.yaml", persist=True)[0]
        raw_predictions = self.entity_model.predict(frame_bgr, verbose=False, conf=self.args["threshold"], iou=self.args["iou_threshold"])[0]
        outputs = []

        for detection in raw_predictions.boxes:
            outputs.append({
                "label": self.args["labels"][int(detection.cls)],
                "confidence": float(detection.conf),
                "bbox": detection.xyxyn[0],
                #"id": int(detection.id)
                
                "health": None,
            })

        return outputs
    def augment_predictions(self, frame_bgr, frame_rgb, raw_predictions):
        for entity in raw_predictions:
            label = entity['label']
            bbox = entity['bbox']

            x1 = int(bbox[0] * frame_bgr.shape[1])
            y1 = int(bbox[1] * frame_bgr.shape[0])
            x2 = int(bbox[2] * frame_bgr.shape[1])
            y2 = int(bbox[3] * frame_bgr.shape[0])


            if label in ["Enemy", "Friendly", "Me"]:
                entity_height = y2 - y1
                half_height = entity_height // 2
                cropped_y2 = y1 + half_height

                roi_bb_extra = 8
                roi = frame_rgb[y1-roi_bb_extra:cropped_y2+roi_bb_extra+1, x1-roi_bb_extra:x2+roi_bb_extra+1] 

                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue

                #detect_start = time.time()
                ocr_detections = self.ocr.detect(roi)
                #print((time.time() - detect_start) * 1000, "detect")


                health = None

                ocr_detections = ocr_detections[0]
                for ocr_val in ocr_detections:
                    for bounding_box in ocr_val:
                        if len(bounding_box) == 0:
                            continue

                        x1_ocr = int(bounding_box[0])
                        x2_ocr = int(bounding_box[1])
                        y1_ocr = int(bounding_box[2])
                        y2_ocr = int(bounding_box[3])

                        ocr_roi = roi[y1_ocr:y2_ocr+1, x1_ocr:x2_ocr+1]
                        if ocr_roi.shape[0] == 0 or ocr_roi.shape[1] == 0:
                            continue

                        white_mask = (
                            (ocr_roi[:, :, 0] > 150) & 
                            (ocr_roi[:, :, 1] > 150) & 
                            (ocr_roi[:, :, 2] > 150)
                        )

                        #white_mask_8bit = (white_mask.astype(np.uint8)) * 255

                        gray_roi = cv2.cvtColor(ocr_roi, cv2.COLOR_BGR2GRAY) * white_mask.astype(np.uint8)

                        white_pixel_count = np.sum(white_mask)
                        total_pixel_count = ocr_roi.shape[0] * ocr_roi.shape[1]
                        white_percentage = white_pixel_count / total_pixel_count

                        if white_percentage > 0.1:
                            #filename = f"./random/white_mask_{uuid.uuid4().hex[:8]}.png"
                            #cv2.imwrite(filename, gray_roi)

                            #cv2.rectangle(frame_bgr, (x1 + x1_ocr - 8, y1 + y1_ocr - 8), (x1 + x2_ocr - 8, y1 + y2_ocr - 8), (255, 255, 255), 2)



                            # Health text


                            #recon_start = time.time()
                            text = self.ocr.recognize(
                                gray_roi, 
                                detail=False,
                                paragraph=True,
                                contrast_ths=0,
                            )[0]
                            #print((time.time() - recon_start) * 1000, "recon")

                            health = transform_string_to_number(text)
                            entity["health"] = health

                            #recon_start = time.time()

                            """tesseract_output = pytesseract.image_to_string(
                                gray_roi,
                                config="--psm 7 digits",
                            )  

                            #print((time.time() - recon_start) * 1000, "recon")

                            health = transform_string_to_number(tesseract_output)"""


                            """# Health procentage

                            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV_FULL)
                            mask_target = health_values[label]
                            roi_health_procentage_mask = cv2.inRange(roi_hsv, mask_target["lower_bound"], mask_target["upper_bound"])

                            #file_id = f"./random/{uuid.uuid4().hex[:8]}"
                            #cv2.imwrite(file_id + "mask.png", roi_health_procentage_mask)
                            #cv2.imwrite(file_id + "raw.png", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

                            start_pixel_x = int((x1_ocr - roi_bb_extra + x2_ocr - roi_bb_extra) / 2)
                            procentage_pixel_y = y2_ocr - roi_bb_extra #int((y1_ocr + y2_ocr) / 2) + 6#int((y2_ocr - y1_ocr) * 0.2) # 20% of the difference

                            chosen_x1 = 0
                            chosen_x2 = 0
                            biggest_difference = 0

                            for y in range(procentage_pixel_y - 8, procentage_pixel_y + 9):
                                if y < 0 or y >= ocr_roi.shape[0]:
                                    break


                                smallest_1_x = roi_hsv.shape[1]
                                biggest_2_x = 0

                                for x in range(0, start_pixel_x):
                                    color_percentage = roi_health_procentage_mask[y, x]

                                    if color_percentage:
                                        if(x < smallest_1_x):
                                            smallest_1_x = x

                                for x in range(start_pixel_x, roi_hsv.shape[1] - 1):
                                    color_percentage = roi_health_procentage_mask[y, x]

                                    if color_percentage:
                                        if(x > biggest_2_x):
                                            biggest_2_x = x

                                difference = biggest_2_x - smallest_1_x
                                if difference > biggest_difference:
                                    biggest_difference = difference
                                    chosen_x1 = smallest_1_x
                                    chosen_x2 = biggest_2_x


                            cv2.rectangle(frame_bgr, (x1 + chosen_x1, y1 + procentage_pixel_y), (x1 + chosen_x2, y1 + procentage_pixel_y + 2), (255, 255, 255), 2)"""

        return raw_predictions