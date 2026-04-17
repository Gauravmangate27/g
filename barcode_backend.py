from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception:
    zbar_decode = None

app = Flask(__name__)


def extract_code_value(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    digit_groups = re.findall(r"\d+", text)
    if digit_groups:
        return max(digit_groups, key=len)
    return text


def decode_image(image: np.ndarray) -> Optional[dict]:
    if image is None or image.size == 0:
        return None

    qr_detector = cv2.QRCodeDetector()

    # Try OpenCV single QR decode first, then multi-code fallback.
    single_text, _, _ = qr_detector.detectAndDecode(image)
    if single_text:
        return {
            "format": "QR_CODE",
            "text": single_text,
            "numeric": extract_code_value(single_text),
        }

    ok, decoded_list, _, _ = qr_detector.detectAndDecodeMulti(image)
    if ok and decoded_list:
        for text in decoded_list:
            if text:
                return {
                    "format": "QR_CODE",
                    "text": text,
                    "numeric": extract_code_value(text),
                }

    # Then try pyzbar for 1D and 2D barcodes when available.
    if zbar_decode is not None:
        decoded = zbar_decode(image)
        for item in decoded:
            try:
                text = item.data.decode("utf-8", errors="replace").strip()
            except Exception:
                text = ""
            if text:
                return {
                    "format": item.type,
                    "text": text,
                    "numeric": extract_code_value(text),
                }

    return None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/decode", methods=["POST", "OPTIONS"])
def decode_route():
    if request.method == "OPTIONS":
        return ("", 204)

    file = request.files.get("image")
    if not file:
        return jsonify({"found": False, "error": "Missing image file"}), 400

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"found": False, "error": "Empty image file"}), 400

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    result = decode_image(image)

    if not result:
        return jsonify({"found": False})

    return jsonify(
        {
            "found": True,
            "format": result["format"],
            "text": result["text"],
            "numeric": result["numeric"],
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
