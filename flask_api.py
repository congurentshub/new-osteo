# ============================================================
#  OsteoAI — Flask REST API
#  Extracted from APPR.py (Streamlit) by Claude
#
#  Install dependencies:
#    pip install flask flask-cors torch torchvision pillow
#                opencv-python fpdf werkzeug numpy requests
#
#  Run:
#    python flask_api.py
#
#  All endpoints are CORS-enabled so your GitHub Pages
#  frontend can call them directly.
# ============================================================

import os
import io
import sqlite3
import base64
import json
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from fpdf import FPDF
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ─────────────────────────────────────────────
#  CONFIG  (edit these paths to match your setup)
# ─────────────────────────────────────────────
DB_PATH    = "database.db"
MODEL_PATH = os.path.join("models", "E6_albumentations.pth")
TMP_DIR    = "tmp"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES    = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

os.makedirs(TMP_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)   # Allow all origins — restrict in production if needed


# ════════════════════════════════════════════════════════════
#  DATABASE  (SQLite — same schema as your original app)
# ════════════════════════════════════════════════════════════

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name     TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  TEXT UNIQUE,
            name        TEXT,
            age         INTEGER,
            gender      TEXT,
            last_visit  TEXT,
            notes       TEXT,
            created_by  INTEGER,
            FOREIGN KEY (created_by) REFERENCES users(id)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      TEXT,
            predicted_grade TEXT,
            timestamp       TEXT,
            user_id         INTEGER,
            orig_image_path TEXT,
            heatmap_path    TEXT,
            notes           TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    conn.commit()
    conn.close()


# ── Auth helpers ──────────────────────────────────────────

def register_user(username, password, full_name=""):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        pw_hash = generate_password_hash(password)
        cur.execute(
            "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
            (username, pw_hash, full_name)
        )
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and check_password_hash(row["password_hash"], password):
        return {"id": row["id"], "username": row["username"], "full_name": row["full_name"]}
    return None


# ── Patient helpers ───────────────────────────────────────

def create_patient(patient_id, name, age, gender, last_visit, notes, created_by):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO patients (patient_id, name, age, gender, last_visit, notes, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (patient_id, name, age, gender, last_visit, notes, created_by))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "Patient ID already exists"
    finally:
        conn.close()


def read_patients(limit=200):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT p.*, u.username AS created_by_username
        FROM patients p
        LEFT JOIN users u ON p.created_by = u.id
        ORDER BY p.id DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_patient(row_id, **fields):
    conn = get_connection()
    cur  = conn.cursor()
    keys = ", ".join([f"{k} = ?" for k in fields.keys()])
    vals = list(fields.values()) + [row_id]
    cur.execute(f"UPDATE patients SET {keys} WHERE id = ?", vals)
    conn.commit()
    conn.close()


def delete_patient(row_id):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM patients WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


def get_patient_by_patient_id(patient_id):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# ── Inference-log helpers ─────────────────────────────────

def log_inference(patient_id, predicted_grade, user_id,
                  orig_image_path, heatmap_path, notes=""):
    conn = get_connection()
    cur  = conn.cursor()
    timestamp = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO inference_logs
            (patient_id, predicted_grade, timestamp, user_id,
             orig_image_path, heatmap_path, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, predicted_grade, timestamp,
          user_id, orig_image_path, heatmap_path, notes))
    conn.commit()
    conn.close()


def read_inference_logs(limit=100):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT il.*, u.username AS user_name
        FROM inference_logs il
        LEFT JOIN users u ON il.user_id = u.id
        ORDER BY il.id DESC LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════
#  MODEL  (ResNet50 — loaded once at startup)
# ════════════════════════════════════════════════════════════

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found at {MODEL_PATH}. /predict will be disabled.")
        return None
    base = models.resnet50(pretrained=False)
    base.fc = nn.Linear(base.fc.in_features, len(CLASSES))
    base.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    base.to(DEVICE).eval()
    print(f"[INFO] Model loaded from {MODEL_PATH} on {DEVICE}")
    return base


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

MODEL = load_model()


# ════════════════════════════════════════════════════════════
#  GRAD-CAM  (exact same logic as your Streamlit app)
# ════════════════════════════════════════════════════════════

def generate_gradcam(model, x_tensor):
    """
    Returns (heatmap_rgb_np_array, class_index)
    heatmap is already overlaid on the resized original image.
    """
    model.eval()
    feature_map = []
    gradient    = []

    def forward_hook(module, inp, out):
        feature_map.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradient.append(grad_out[0])

    # Target the last conv block in layer4 (ResNet50)
    target_layer = model.layer4[-1]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    x   = x_tensor.clone().requires_grad_(True)
    out = model(x)
    cls = out.argmax(dim=1).item()
    model.zero_grad()
    out[0, cls].backward()

    h1.remove()
    h2.remove()

    if not gradient or not feature_map:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        return blank, cls

    pooled_grads = gradient[0].mean(dim=[2, 3], keepdim=True)
    feat = feature_map[0]
    cam  = (feat * pooled_grads).sum(dim=1, keepdim=False).squeeze().cpu().detach().numpy()
    cam  = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam     = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap, cls


# ════════════════════════════════════════════════════════════
#  PDF REPORT GENERATOR
# ════════════════════════════════════════════════════════════

def generate_pdf_report(predicted_grade, orig_path, heatmap_path, patient_info=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="OsteoAI Detection Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    if patient_info:
        pdf.cell(200, 10, txt=f"Patient ID : {patient_info.get('patient_id', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Name       : {patient_info.get('name', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Age        : {patient_info.get('age', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Gender     : {patient_info.get('gender', 'N/A')}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Grade : {predicted_grade}", ln=True)
    pdf.cell(200, 10, txt=f"Timestamp       : {datetime.utcnow().isoformat()}", ln=True)
    pdf.ln(10)
    if os.path.exists(orig_path):
        pdf.image(orig_path, x=10, y=None, w=90)
    if os.path.exists(heatmap_path):
        pdf.image(heatmap_path, x=110, y=None, w=90)
    out_path = os.path.join(TMP_DIR, f"report_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf")
    pdf.output(out_path)
    return out_path


# ════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ════════════════════════════════════════════════════════════

# ── Health check ─────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":        "OsteoAI Flask API is running",
        "model_loaded":  MODEL is not None,
        "device":        DEVICE
    })


# ── Auth routes ───────────────────────────────────────────

@app.route("/auth/register", methods=["POST"])
def route_register():
    """
    POST /auth/register
    Body: { "username": "...", "password": "...", "full_name": "..." }
    """
    data      = request.get_json()
    username  = data.get("username", "").strip()
    password  = data.get("password", "").strip()
    full_name = data.get("full_name", "").strip()

    if not username or not password:
        return jsonify({"error": "username and password required"}), 400

    ok, err = register_user(username, password, full_name)
    if ok:
        return jsonify({"message": "User registered successfully"})
    return jsonify({"error": err}), 409


@app.route("/auth/login", methods=["POST"])
def route_login():
    """
    POST /auth/login
    Body: { "username": "...", "password": "..." }
    Returns: { "user": { "id": ..., "username": ..., "full_name": ... } }
    """
    data     = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    user = authenticate_user(username, password)
    if user:
        return jsonify({"user": user})
    return jsonify({"error": "Invalid credentials"}), 401


# ── Patient routes ────────────────────────────────────────

@app.route("/patients", methods=["GET"])
def route_get_patients():
    """GET /patients?limit=200"""
    limit = int(request.args.get("limit", 200))
    return jsonify(read_patients(limit))


@app.route("/patients", methods=["POST"])
def route_create_patient():
    """
    POST /patients
    Body: { "patient_id","name","age","gender","last_visit","notes","created_by" }
    """
    d  = request.get_json()
    ok, err = create_patient(
        d.get("patient_id"), d.get("name"), int(d.get("age", 0)),
        d.get("gender"), d.get("last_visit"), d.get("notes", ""),
        d.get("created_by")
    )
    if ok:
        return jsonify({"message": "Patient created"})
    return jsonify({"error": err}), 409


@app.route("/patients/<int:row_id>", methods=["PUT"])
def route_update_patient(row_id):
    """
    PUT /patients/<id>
    Body: any subset of { name, age, gender, last_visit, notes }
    """
    d = request.get_json()
    allowed = {"name", "age", "gender", "last_visit", "notes"}
    fields  = {k: v for k, v in d.items() if k in allowed}
    update_patient(row_id, **fields)
    return jsonify({"message": "Patient updated"})


@app.route("/patients/<int:row_id>", methods=["DELETE"])
def route_delete_patient(row_id):
    """DELETE /patients/<id>"""
    delete_patient(row_id)
    return jsonify({"message": "Patient deleted"})


@app.route("/patients/by-pid/<patient_id>", methods=["GET"])
def route_get_patient_by_pid(patient_id):
    """GET /patients/by-pid/<patient_id>"""
    p = get_patient_by_patient_id(patient_id)
    if p:
        return jsonify(p)
    return jsonify({"error": "Not found"}), 404


# ── Inference-log routes ──────────────────────────────────

@app.route("/logs", methods=["GET"])
def route_get_logs():
    """GET /logs?limit=100"""
    limit = int(request.args.get("limit", 100))
    return jsonify(read_inference_logs(limit))


# ── Prediction + Grad-CAM route ───────────────────────────

@app.route("/predict", methods=["POST"])
def route_predict():
    """
    POST /predict  (multipart/form-data)
    Fields:
      - image      : the X-ray file (required)
      - patient_id : string (optional — if provided, logs are saved)
      - user_id    : integer (optional)
      - notes      : string (optional)

    Returns JSON:
    {
      "grade"         : "Grade 2",
      "grade_index"   : 2,
      "overlay_b64"   : "<base64-encoded PNG of the Grad-CAM overlay>",
      "orig_path"     : "tmp/orig_....jpg",   (if saved)
      "heatmap_path"  : "tmp/heat_....jpg",   (if saved)
      "log_saved"     : true | false
    }
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Check MODEL_PATH."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    img  = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Run model + Grad-CAM
    x_tensor         = transform(img).unsqueeze(0).to(DEVICE)
    heatmap, cls     = generate_gradcam(MODEL, x_tensor)
    grade            = CLASSES[cls]

    # Build overlay
    img_np  = np.array(img.resize((224, 224)))
    overlay = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)

    # Encode overlay to base64 PNG so the frontend can display it directly
    _, buffer   = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buffer).decode("utf-8")

    # Optional: save to disk and log
    patient_id  = request.form.get("patient_id")
    user_id     = request.form.get("user_id")
    notes       = request.form.get("notes", "")
    log_saved   = False
    orig_path   = ""
    heat_path   = ""

    if patient_id and user_id:
        ts        = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        orig_path = os.path.join(TMP_DIR, f"orig_{ts}.jpg")
        heat_path = os.path.join(TMP_DIR, f"heat_{ts}.jpg")
        img.save(orig_path)
        cv2.imwrite(heat_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        log_inference(patient_id, grade, int(user_id), orig_path, heat_path, notes)
        log_saved = True

    return jsonify({
        "grade":        grade,
        "grade_index":  cls,
        "overlay_b64":  overlay_b64,
        "orig_path":    orig_path,
        "heatmap_path": heat_path,
        "log_saved":    log_saved
    })


# ── PDF report route ──────────────────────────────────────

@app.route("/report/pdf", methods=["POST"])
def route_generate_pdf():
    """
    POST /report/pdf
    Body: {
      "predicted_grade" : "Grade 2",
      "orig_path"       : "tmp/orig_....jpg",
      "heatmap_path"    : "tmp/heat_....jpg",
      "patient_info"    : { "patient_id": "...", "name": "...", "age": ..., "gender": "..." }
    }
    Returns the PDF file as a download.
    """
    d = request.get_json()
    pdf_path = generate_pdf_report(
        predicted_grade = d.get("predicted_grade", "Unknown"),
        orig_path       = d.get("orig_path", ""),
        heatmap_path    = d.get("heatmap_path", ""),
        patient_info    = d.get("patient_info")
    )
    return send_file(pdf_path, as_attachment=True,
                     download_name="osteoai_report.pdf",
                     mimetype="application/pdf")


# ── Groq AI chat proxy ────────────────────────────────────

@app.route("/chat", methods=["POST"])
def route_chat():
    """
    POST /chat
    Body: {
      "api_key"  : "gsk_...",
      "messages" : [ {"role": "user", "content": "..."}, ... ],
      "xray_context": "Grade 2 for patient P001"   (optional)
    }
    Proxies to Groq so your API key stays server-side (safer).
    """
    import requests as req

    d       = request.get_json()
    api_key = d.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "api_key required"}), 400

    system_messages = [{
        "role": "system",
        "content": (
            "You are an expert medical AI assistant specialising in osteoporosis and "
            "bone health. Provide accurate, compassionate, and professional medical "
            "information. Always remind users to consult healthcare professionals for "
            "personalised advice. Be clear, concise, and supportive."
        )
    }]

    if d.get("xray_context"):
        system_messages.append({
            "role": "system",
            "content": f"Current patient context: {d['xray_context']}"
        })

    all_messages = system_messages + d.get("messages", [])

    try:
        resp = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json"
            },
            json={
                "model":       "llama-3.3-70b-versatile",
                "messages":    all_messages,
                "temperature": 0.7,
                "max_tokens":  1024
            },
            timeout=30
        )
        if resp.status_code == 200:
            reply = resp.json()["choices"][0]["message"]["content"]
            return jsonify({"reply": reply})
        error_msg = resp.json().get("error", {}).get("message", "Unknown error")
        return jsonify({"error": error_msg}), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════
#  STARTUP
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    init_db()
    print("[INFO] Database initialised.")
    # Set debug=False and use gunicorn in production (Render/Railway)
    app.run(host="0.0.0.0", port=5000, debug=True)
