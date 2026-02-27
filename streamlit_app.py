import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import time
import re
import difflib
from db import save_upload
import os

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Rotten or Not 🍎", layout="wide")

# --- Glassmorphism dark theme CSS ---
st.markdown(
    """
    <style>
    :root{--glass-bg: rgba(255,255,255,0.04); --glass-border: rgba(255,255,255,0.06);}
    .stApp { background: linear-gradient(180deg,#071028 0%, #081226 100%); color: #e6eef8; }
    .glass {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        box-shadow: 0 8px 32px rgba(2,6,23,0.6);
        backdrop-filter: blur(8px) saturate(140%);
        -webkit-backdrop-filter: blur(8px) saturate(140%);
        border-radius: 14px;
        padding: 18px;
    }
    .big-title{font-size:28px; font-weight:700; color: #f8fafc; margin:0 0 6px 0}
    .subtitle{color:#9aa6b2; margin:0 0 8px 0}
    .stButton>button{background:linear-gradient(90deg,#6ee7b7,#60a5fa); color:#041826; font-weight:700}
    .stFileUploader>div>div, .stFileUploader>div>label{background: rgba(255,255,255,0.02); border-radius:8px}
    .nav-links a{color:#cfeffd; margin:0 10px; text-decoration:none; font-weight:600}
    .nav-links a:hover{color:#7ef0c0; text-decoration:underline}
    .cta-button{background:linear-gradient(90deg,#7ef0c0,#60a5fa); color:#041826; padding:8px 12px; border-radius:8px; font-weight:800; text-decoration:none}
    </style>
    """,
    unsafe_allow_html=True,
)



# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best1.pt")

model = load_model()


# Single-language (English) strings
STRINGS = {
    "app_title": "🍓 Fruit Freshness Detector",
    "app_subtitle": "Detect whether a fruit is fresh or rotten using YOLO",
    "upload_header": "📤 Upload Fruit Image",
    "upload_label": "Upload Image",
    "uploaded_caption": "Uploaded Image",
    "detection_caption": "Detection Result",
    "no_fruit": "⚠️ No fruit detected.",
    "recipes_header": "Recipe Ideas",
    "no_recipe_for": "No recipe found for {name}.",
    "model_loaded": "✅ Model loaded successfully!",
    "detection_details": "Detection details",
    "select_recipe": "Select fruit for recipe (override)",
    "auto_map": "Auto-select best match",
    "confidence_threshold": "Confidence threshold",
    "auto_map_info": "Auto-mapping uses label normalization, substring and fuzzy match.",
    "auto_map_failed": "Auto-mapping couldn't find a good match; please select manually.",
}


def t(key, **kwargs):
    text = STRINGS.get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text

# Sidebar: language + quick controls (glass card)
with st.sidebar:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.success(t("model_loaded"))
    st.markdown("---")
    st.markdown("**Controls**")
    auto = st.checkbox(t("auto_map"), value=True)
    conf_thresh = st.slider(t("confidence_threshold"), 0.0, 1.0, 0.3, 0.05)
    st.markdown("</div>", unsafe_allow_html=True)

# show translated title/subtitle (hero)
st.markdown("<div class='glass' style='margin-bottom:18px'><div class='big-title'>" + t("app_title") + "</div><div class='subtitle'>" + t("app_subtitle") + "</div></div>", unsafe_allow_html=True)

# --- Top navbar (glass) ---
st.markdown(
        """
        <div class='glass' style='display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;padding:12px 18px'>
            <div style='display:flex;align-items:center;gap:12px'>
                <div style='width:44px;height:44px;border-radius:10px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#60a5fa22,#7ef0c022);font-weight:800'>🍏</div>
                <div style='font-weight:700;color:#f8fafc'>Rotten or Not</div>
            </div>
            <div style='display:flex;align-items:center;gap:18px'>
                <div class='nav-links'>
                    <a href="#upload">""" + t("upload_header") + """</a>
                    <a href="#recipes">""" + t("recipes_header") + """</a>
                    <a href="#about">About</a>
                </div>
                <a class='cta-button' href="#upload">""" + "Try it" + """</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
)

# Simple recipe database (extend as needed)
RECIPES = {
    "apple": {
        "title": "Apple Crumble",
        "content": "Ingredients:\n- 4 apples\n- 100g flour\n- 75g butter\n- 75g brown sugar\n\nSteps:\n1. Slice apples and place in a baking dish.\n2. Mix flour, butter and sugar into crumbs and sprinkle over apples.\n3. Bake at 180°C for 30-35 minutes until golden."
    },
    "banana": {
        "title": "Banana Smoothie",
        "content": "Ingredients:\n- 2 ripe bananas\n- 250ml milk (or plant milk)\n- 1 tbsp honey\n\nSteps:\n1. Blend all ingredients until smooth.\n2. Serve chilled."
    },
    "mango": {
        "title": "Mango Salsa",
        "content": "Ingredients:\n- 1 ripe mango\n- 1/2 red onion\n- Juice of 1 lime\n- Handful cilantro\n\nSteps:\n1. Dice mango and onion.\n2. Mix with lime juice and chopped cilantro.\n3. Serve with chips or grilled fish."
    },
    "orange": {
        "title": "Orange Granita",
        "content": "Ingredients:\n- 500ml fresh orange juice\n- 50g sugar\n\nSteps:\n1. Dissolve sugar into juice.\n2. Freeze in a shallow tray, scraping every 30 minutes until flaky."
    },
    "strawberry": {
        "title": "Strawberry Salad",
        "content": "Ingredients:\n- 250g strawberries\n- Handful of spinach\n- Balsamic vinaigrette\n\nSteps:\n1. Halve strawberries and toss with spinach.\n2. Drizzle with vinaigrette and serve."
    }
    ,
    "cucumber": {
        "title": "Cucumber Raita",
        "content": "Ingredients:\n- 1 large cucumber\n- 250g plain yogurt\n- 1/2 tsp roasted cumin powder\n- Salt to taste\n- Fresh cilantro or mint (optional)\n\nSteps:\n1. Peel and grate or finely chop the cucumber.\n2. Mix cucumber with yogurt, cumin powder and salt.\n3. Garnish with chopped cilantro or mint and serve chilled as a side."
    }
}

# (Multilingual recipe translations removed — site is English-only)

def extract_fruit_name(label: str) -> str:
    """Normalize model label to a fruit name key used in RECIPES."""
    s = label.lower()
    s = s.replace("_", " ")
    # remove words indicating freshness
    s = re.sub(r"\b(fresh|rotten|ripe|unripe|good|bad)\b", "", s)
    s = re.sub(r"[^a-z\s]", "", s)
    s = s.strip()
    # if label contains multiple words, pick the last as likely fruit (common model patterns)
    parts = s.split()
    if len(parts) == 0:
        return ""
    # try to find a known fruit in parts
    for p in parts:
        if p in RECIPES:
            return p
    # fallback to last token
    return parts[-1]


def auto_map_fruit(detected_info, conf_thresh=0.3):
    """Try to auto-map model detections to a known recipe key.

    Strategy (in order of checking per detection sorted by confidence):
    - Normalize label and check exact recipe key
    - Check if any recipe key is substring of label
    - Fuzzy match label against recipe keys using difflib
    Returns the first reasonable match or None.
    """
    if not detected_info:
        return None

    # sort by confidence desc
    items = sorted(detected_info, key=lambda x: x.get("conf", 0), reverse=True)
    keys = list(RECIPES.keys())

    for it in items:
        conf = float(it.get("conf", 0))
        if conf < conf_thresh:
            continue
        label = it.get("label", "").lower()
        name = extract_fruit_name(label)
        if name in RECIPES:
            return name
        # substring
        for k in keys:
            if k in label:
                return k
        # fuzzy match against full label
        match = difflib.get_close_matches(label, keys, n=1, cutoff=0.6)
        if match:
            return match[0]
        # try tokens
        for token in label.split():
            match = difflib.get_close_matches(token, keys, n=1, cutoff=0.7)
            if match:
                return match[0]

    return None

# ===
# =====================================================
st.markdown("<div class='glass' style='margin-bottom:12px;padding:12px'><h2 id='upload' style='margin:0;color:#f8fafc'>" + t("upload_header") + "</h2></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    t("upload_label"),
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # read raw bytes once so we can both decode and save them
    raw_bytes = uploaded_file.read()
    file_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # resize large images
    frame_resized = cv2.resize(frame_rgb, (640, 640))

    st.image(frame_rgb, caption=t("uploaded_caption"), width="stretch")

    results = model.predict(frame_resized, conf=0.5, verbose=False)
    pred = results[0]

    if pred.boxes is not None and len(pred.boxes) > 0:
        detected_labels = []
        detected_info = []
        for box in pred.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = pred.names[cls_id]
            detected_labels.append(label)
            detected_info.append({"label": label, "conf": float(conf), "cls_id": int(cls_id)})

            color = (0,255,0) if "fresh" in label.lower() else (0,0,255)

            cv2.rectangle(frame_resized,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame_resized,
                        f"{label} {conf:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

        st.image(frame_resized,
             caption=t("detection_caption"),
             width="stretch")

        # Show detection details and allow manual override for recipe selection
        st.markdown("---")
        if t("detection_details", ):
            pass
        with st.expander(t("detection_details")):
            st.write("Detected labels and confidences:")
            st.write(detected_info)
            st.write("Model class mapping (id -> name):")
            try:
                st.write(pred.names)
            except Exception:
                st.write("(no mapping available)")

        # Auto-mapping info (controls moved to sidebar)
        st.markdown(":information_source: " + t("auto_map_info"))

        options = sorted(RECIPES.keys())
        chosen_fruit = None
        if auto:
            auto_choice = auto_map_fruit(detected_info, conf_thresh=conf_thresh)
            if auto_choice:
                chosen_fruit = auto_choice
                st.success(f"Auto-selected: {chosen_fruit}")
            else:
                st.warning(t("auto_map_failed"))

        # If not auto-selected, show manual selector (default to first detected normalized)
        if not chosen_fruit:
            # Build default selection (first normalized detected fruit if any)
            fruit_keys = []
            for lab in detected_labels:
                name = extract_fruit_name(lab)
                if name:
                    fruit_keys.append(name)

            default_idx = 0
            if len(fruit_keys) > 0 and fruit_keys[0] in options:
                default_idx = options.index(fruit_keys[0])

            chosen_fruit = st.selectbox(t("select_recipe"), options, index=default_idx)

        st.markdown("<div class='glass' style='margin-top:12px;padding:12px'><h3 id='recipes' style='margin:0;color:#f8fafc'>" + t("recipes_header") + "</h3></div>", unsafe_allow_html=True)
        if chosen_fruit in RECIPES:
            r = RECIPES[chosen_fruit]
            st.subheader(r.get("title", chosen_fruit.title()))
            st.text(r.get("content", ""))
        else:
            st.info(STRINGS.get("no_recipe_for").format(name=chosen_fruit))

        # Try saving upload + detection metadata to MongoDB and Cloudinary (non-fatal)
        try:
            cloud_cfg = {
                "cloud_name": os.getenv("CLOUDINARY_CLOUD_NAME", "dgosjbdx7"),
                "api_key": os.getenv("CLOUDINARY_API_KEY", "764318225397556"),
                "api_secret": os.getenv("CLOUDINARY_API_SECRET", "2_tKwqV7ZpG0d-nfgADM6jBXHnQ"),
            }
            save_res = save_upload(raw_bytes, getattr(uploaded_file, "name", "upload"), chosen_fruit, detected_info, cloudinary_config=cloud_cfg)
            st.caption(f"Saved upload to database: {str(save_res.get('_id'))}")
            if save_res.get("cloudinary"):
                st.markdown(f"Uploaded to Cloudinary: {save_res['cloudinary'].get('secure_url')}")
        except Exception as e:
            st.warning(f"Could not save upload to database/cloud: {e}")

    else:
        st.warning("⚠️ No fruit detected.")


