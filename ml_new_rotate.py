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
from paddleocr import PaddleOCR

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
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=torch.cuda.is_available(), show_log=False)
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
            self.lpr_model.load_state_dict(checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint)
            self.lpr_model.to(self.device)
            self.lpr_model.eval()

    def detect_plates(self, image):
        h, w = image.shape[:2]
        plates = []
        if self.lpd_model is None: return plates
        with self.inference_lock:
            results = self.lpd_model(image, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                pad = 30
                x1, y1, x2, y2 = max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)
                plates.append({"confidence": conf, "cropped_image": image[y1:y2, x1:x2], "bbox": [x1, y1, x2, y2]})
        return plates

    def recognize(self, cropped_image):
        if cropped_image is None: return "", ([], [])
        corrected = self._correct_rotation_safe(cropped_image)
        
        # 1. Try PARSeq first
        resized = cv2.resize(corrected, (128, 32))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            logits = self.lpr_model(tensor, max_length=15)
            tokens, probs = self.lpr_model.tokenizer.decode(logits.softmax(-1))
        
        parseq_text = tokens[0]
        parseq_conf = np.mean(probs[0])

        # 2. Fallback to PaddleOCR if PARSeq is unsure or text is too short (typical for failed Hanzi)
        if parseq_conf < 0.8 or len(parseq_text) < 4:
            with self.inference_lock:
                paddle_res = self.paddle_ocr.ocr(corrected, cls=True)
            if paddle_res and paddle_res[0]:
                paddle_text = paddle_res[0][0][1][0]
                paddle_conf = paddle_res[0][0][1][1]
                return paddle_text, (list(paddle_text), [float(paddle_conf)]*len(paddle_text))

        return parseq_text, (list(parseq_text), [float(p) for p in probs[0]])

    def _correct_rotation_safe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return image
        cnt = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            pts = self._order_points(approx.reshape(4, 2))
            w = int(max(np.linalg.norm(pts[2] - pts[3]), np.linalg.norm(pts[1] - pts[0])))
            h = int(max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[0] - pts[3])))
            dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
            return cv2.warpPerspective(image, M, (w, h))
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        if w < h: angle -= 90
        M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.0)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

    def recognize_image(self, image_bytes):
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        plates = self.detect_plates(image)
        results = []
        for p in plates:
            text, (chars, confs) = self.recognize(p["cropped_image"])
            results.append({"text": text, "confidence": p["confidence"], "bbox": p["bbox"], "char_confidence": dict(zip(chars, confs)), "img": cv2.imencode(".jpg", p["cropped_image"])[1].tobytes()})
        return results

ml_service = MLService()
