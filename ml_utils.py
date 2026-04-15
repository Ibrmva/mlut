import torch
import cv2
import numpy as np
import os
import sys
import logging
import yaml
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import threading
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS = os.path.join(BASE_DIR, "artifacts")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ARTIFACTS)

from strhub.models.parseq.system import PARSeq

logger = logging.getLogger(__name__)


class MLService:
    def __init__(self):
        self.ml_executor = ThreadPoolExecutor(max_workers=1)
        self.inference_lock = threading.Lock()
        self.device = torch.device(os.getenv('ML_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.lpd_model = None
        self.lpr_model = None
        self._load_models()

    def _load_models(self):
        lpd_path = "app/artifacts/lpd/license_plate_detector.pt"
        lpr_path = "app/artifacts/lpr/evaluated_ocr_model.pt"
        config_path = "app/configs/model/parseq.yaml"
        charset_path = "app/configs/charset/label.yaml"

        if os.path.exists(lpd_path):
            self.lpd_model = YOLO(lpd_path)
            self.lpd_model.to(self.device)

        if os.path.exists(lpr_path) and os.path.exists(config_path) and os.path.exists(charset_path):
            checkpoint = torch.load(lpr_path, map_location=self.device, weights_only=False)

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            with open(charset_path, "r") as f:
                charset = yaml.safe_load(f)

            self.lpr_model = PARSeq(
                charset_train=charset["model"]["charset_train"],
                charset_test=charset["model"]["charset_train"],
                max_label_length=config.get("max_label_length", 25),
                batch_size=config.get("batch_size", 64),
                lr=config.get("lr", 7e-4),
                warmup_pct=config.get("warmup_pct", 0.075),
                weight_decay=config.get("weight_decay", 0.0),
                img_size=config.get("img_size", [32, 128]),
                patch_size=config.get("patch_size", [4, 8]),
                embed_dim=config.get("embed_dim", 384),
                enc_num_heads=config.get("enc_num_heads", 6),
                enc_mlp_ratio=config.get("enc_mlp_ratio", 4),
                enc_depth=config.get("enc_depth", 12),
                dec_num_heads=config.get("dec_num_heads", 12),
                dec_mlp_ratio=config.get("dec_mlp_ratio", 4),
                dec_depth=config.get("dec_depth", 1),
                perm_num=config.get("perm_num", 6),
                perm_forward=config.get("perm_forward", True),
                perm_mirrored=config.get("perm_mirrored", True),
                decode_ar=config.get("decode_ar", True),
                refine_iters=config.get("refine_iters", 1),
                dropout=config.get("dropout", 0.1)
            )

            self.lpr_model.load_state_dict(
                checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint
            )
            self.lpr_model.to(self.device)
            self.lpr_model.eval()

    def detect_plates(self, image):
        h, w = image.shape[:2]
        plates = []

        if self.lpd_model is None:
            return plates

        with self.inference_lock:
            results = self.lpd_model(image, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                pad = 10
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                crop = image[y1:y2, x1:x2]

                plates.append({
                    "confidence": conf,
                    "cropped_image": crop,
                    "bbox": [x1, y1, x2, y2]
                })

        return plates

    def recognize(self, cropped_image):
        if self.lpr_model is None or cropped_image is None:
            return "", ([], [])

        corrected = self._correct_rotation_safe(cropped_image)
        resized = cv2.resize(corrected, (128, 32))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            with self.inference_lock:
                logits = self.lpr_model(tensor, max_length=12)
                tokens, probs = self.lpr_model.tokenizer.decode(logits.softmax(-1))

        return tokens[0], (list(tokens[0]), [float(p) for p in probs[0]])

    def _correct_rotation_safe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 180)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]

        if angle < -45:
            angle += 90
        if abs(angle) < 6:
            return image

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def recognize_image(self, image_bytes):
        np_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        plates = self.detect_plates(image)

        results = []
        for p in plates:
            text, (chars, confs) = self.recognize(p["cropped_image"])

            results.append({
                "text": text,
                "confidence": p["confidence"],
                "bbox": p["bbox"],
                "char_confidence": dict(zip(chars, confs)),
                "img": cv2.imencode(".jpg", p["cropped_image"])[1].tobytes()
            })

        return results


ml_service = MLService()