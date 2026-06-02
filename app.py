import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(
    page_title="LeafGuard - Yaprak Hastalığı Analiz Sistemi",
    page_icon="🍃",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background-color: #060d06; }
    .block-container { padding: 2rem 3rem; }

    .hero-wrap {
        position: relative;
        text-align: center;
        padding: 3rem 0 2rem;
        margin-bottom: 1rem;
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: 0; left: 50%;
        transform: translateX(-50%);
        width: 600px; height: 200px;
        background: radial-gradient(ellipse, rgba(34,197,94,0.18) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: #f0fdf4;
        letter-spacing: -2px;
        line-height: 1;
        margin-bottom: 0.4rem;
    }
    .hero-title span { color: #22c55e; }
    .hero-sub {
        color: #4ade80;
        font-size: 0.9rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        font-weight: 500;
    }

    .pill-nav {
        display: flex;
        gap: 8px;
        justify-content: center;
        margin-bottom: 2.5rem;
    }
    .pill {
        background: #0f1f0f;
        border: 1px solid #1a3a1a;
        border-radius: 100px;
        padding: 6px 18px;
        font-size: 0.78rem;
        color: #4ade80;
        letter-spacing: 1px;
    }

    .card {
        background: #0a180a;
        border: 1px solid #1a2e1a;
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1rem;
    }
    .card-dark {
        background: #060d06;
        border: 1px solid #162016;
        border-radius: 20px;
        padding: 1.8rem;
    }

    .metric-grid { display: flex; gap: 1rem; margin: 1.5rem 0; }
    .metric-box {
        flex: 1;
        background: #0a180a;
        border: 1px solid #1a3a1a;
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #4ade80;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .metric-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0fdf4;
    }
    .metric-val.green { color: #22c55e; }
    .metric-val.yellow { color: #eab308; }
    .metric-val.red { color: #ef4444; }

    .badge {
        display: inline-block;
        padding: 3px 14px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        margin-top: 6px;
    }
    .badge-hafif  { background: rgba(34,197,94,0.12);  color: #22c55e;  border: 1px solid rgba(34,197,94,0.3); }
    .badge-orta   { background: rgba(234,179,8,0.12);  color: #eab308;  border: 1px solid rgba(234,179,8,0.3); }
    .badge-kritik { background: rgba(239,68,68,0.12);  color: #ef4444;  border: 1px solid rgba(239,68,68,0.3); }

    .action-strip {
        border-left: 3px solid;
        border-radius: 0 12px 12px 0;
        padding: 0.8rem 1.2rem;
        margin: 1rem 0;
        background: #0a180a;
        font-size: 0.9rem;
        color: #d1fae5;
    }
    .action-strip.hafif  { border-color: #22c55e; }
    .action-strip.orta   { border-color: #eab308; }
    .action-strip.kritik { border-color: #ef4444; }

    .sec-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #4ade80;
        margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1a3a1a;
    }

    /* Hastalık bilgi kartı */
    .disease-card {
        background: linear-gradient(135deg, #0a1f0a, #0f2a0f);
        border: 1px solid #1e4a1e;
        border-radius: 20px;
        padding: 1.8rem;
        margin-top: 1.5rem;
    }
    .disease-card h3 {
        font-family: 'Syne', sans-serif;
        font-size: 1.3rem;
        color: #f0fdf4;
        margin-bottom: 1rem;
    }
    .info-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .info-chip {
        background: #0f2a0f;
        border: 1px solid #2a5a2a;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        flex: 1;
        min-width: 140px;
    }
    .info-chip-label {
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4ade80;
        margin-bottom: 4px;
    }
    .info-chip-val {
        font-size: 0.88rem;
        color: #d1fae5;
        font-weight: 500;
    }
    .drug-tag {
        display: inline-block;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.25);
        border-radius: 8px;
        padding: 3px 10px;
        font-size: 0.78rem;
        color: #86efac;
        margin: 3px;
    }
    .tip-box {
        background: rgba(234,179,8,0.06);
        border: 1px solid rgba(234,179,8,0.2);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #fef08a;
        margin-top: 0.8rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #0a180a;
        border-radius: 14px;
        padding: 4px;
        border: 1px solid #1a3a1a;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.88rem;
        color: #4ade80;
    }
    .stTabs [aria-selected="true"] {
        background: #14532d !important;
        color: #f0fdf4 !important;
    }

    div[data-testid="stImage"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #1a3a1a;
    }

    .footer {
        text-align: center;
        color: #1a3a1a;
        font-size: 0.75rem;
        margin-top: 4rem;
        padding: 1.5rem;
        border-top: 1px solid #0f2a0f;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Hastalık Bilgi Veritabanı ----
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "neden": "Xanthomonas campestris bakterisi",
        "belirti": "Yapraklarda koyu kahve, suyla çevrili lekeler; meyve üzerinde kabarcıklar",
        "yayilma": "Yağmur sıçrantısı, kontamine tohumlar, böcekler",
        "ilaclar": ["Bakır hidroksit", "Bakır oksiklorür", "Mancozeb", "Streptomisin"],
        "onlem": "Dayanıklı çeşit seçin, sulama suyunun yapraklara değmesini önleyin",
        "aciliyet": "orta"
    },
    "Potato___Early_blight": {
        "neden": "Alternaria solani mantarı",
        "belirti": "Yapraklarda halkalı koyu kahve lekeler (hedef tahtası görünümü)",
        "yayilma": "Rüzgar, yağmur, hasta bitki artıkları",
        "ilaclar": ["Chlorothalonil", "Mancozeb", "Azoxystrobin", "Difenoconazole"],
        "onlem": "Bitki artıklarını yok edin, nöbetleşe ekim yapın",
        "aciliyet": "orta"
    },
    "Potato___Late_blight": {
        "neden": "Phytophthora infestans su küfü",
        "belirti": "Yapraklarda sulamsı koyu lekeler, alt yüzde beyaz küf, çürüme",
        "yayilma": "Rüzgar, yağmur; serin ve nemli hava hızlandırır",
        "ilaclar": ["Metalaxyl", "Cymoxanil", "Fosetyl-Al", "Mancozeb + Cymoxanil"],
        "onlem": "Sertifikalı tohum kullanın, havalandırmayı artırın",
        "aciliyet": "kritik"
    },
    "Tomato_Bacterial_spot": {
        "neden": "Xanthomonas vesicatoria bakterisi",
        "belirti": "Yaprak ve meyvelerde küçük sarımsı-kahve lekeler, zamanla kuruma",
        "yayilma": "Yağmur sıçrantısı, kontamine tohumlar",
        "ilaclar": ["Bakır sülfat", "Bakır hidroksit", "Mancozeb", "Streptomisin"],
        "onlem": "Tohumları ilaçlayın, damla sulama kullanın",
        "aciliyet": "orta"
    },
    "Tomato_Early_blight": {
        "neden": "Alternaria solani mantarı",
        "belirti": "Alt yapraklarda halka halka koyu lekeler, sararma",
        "yayilma": "Hasta bitki artıkları, rüzgar, yağmur sıçrantısı",
        "ilaclar": ["Chlorothalonil", "Azoxystrobin", "Iprodione", "Tebuconazole"],
        "onlem": "Alt yaprakları temizleyin, aşırı azot vermekten kaçının",
        "aciliyet": "orta"
    },
    "Tomato_Late_blight": {
        "neden": "Phytophthora infestans su küfü",
        "belirti": "Yaprak ve gövdelerde sulamsı lekeler, hızlı çürüme",
        "yayilma": "Rüzgar; serin nemli hava kritik",
        "ilaclar": ["Metalaxyl + Mancozeb", "Cymoxanil", "Fosetyl-Al", "Dimethomorph"],
        "onlem": "Dayanıklı çeşit seçin, gece sulamaktan kaçının",
        "aciliyet": "kritik"
    },
    "Tomato_Leaf_Mold": {
        "neden": "Passalora fulva mantarı",
        "belirti": "Yaprak üstünde sarı lekeler, alt yüzde kadifemsi kahve küf",
        "yayilma": "Hava yoluyla sporlar; yüksek nem ortamı",
        "ilaclar": ["Chlorothalonil", "Mancozeb", "Copper-based", "Myclobutanil"],
        "onlem": "Sera havalandırmasını artırın, yaprakların ıslanmasını önleyin",
        "aciliyet": "orta"
    },
    "Tomato_Septoria_leaf_spot": {
        "neden": "Septoria lycopersici mantarı",
        "belirti": "Küçük yuvarlak lekeler: gri merkez, koyu kenar",
        "yayilma": "Yağmur sıçrantısı, hasta bitki artıkları",
        "ilaclar": ["Chlorothalonil", "Mancozeb", "Copper hydroxide", "Azoxystrobin"],
        "onlem": "Alt yaprakları budayın, mulch kullanın",
        "aciliyet": "hafif"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "neden": "Tetranychus urticae (kırmızı örümcek akarı)",
        "belirti": "Yapraklarda sarı benekler, altında ince ağsı, kuruma",
        "yayilma": "Rüzgar, kontamine araç-gereç; kuru sıcak hava hızlandırır",
        "ilaclar": ["Abamectin", "Bifenazate", "Spiromesifen", "Hexythiazox"],
        "onlem": "Düzenli su sisi uygulayın, doğal düşmanları (Phytoseiid) koruyun",
        "aciliyet": "orta"
    },
    "Tomato__Target_Spot": {
        "neden": "Corynespora cassiicola mantarı",
        "belirti": "Yapraklarda halkalı koyu lekeler, zamanla sarı hale",
        "yayilma": "Rüzgar, yağmur; yüksek nem",
        "ilaclar": ["Chlorothalonil", "Azoxystrobin", "Fluxapyroxad", "Tebuconazole"],
        "onlem": "Bitki sıklığını azaltın, hava sirkülasyonunu artırın",
        "aciliyet": "orta"
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "neden": "TYLCV virüsü (Bemisia tabaci beyazsinek taşır)",
        "belirti": "Yapraklar yukarı kıvrılır, sararır, bodurlaşma",
        "yayilma": "Beyazsinek (Bemisia tabaci) vektörü",
        "ilaclar": ["İmidakloprid (vektör kontrolü)", "Thiamethoxam", "Pymetrozine"],
        "onlem": "Sarı yapışkan tuzak kullanın, reflektif malç, dayanıklı çeşit",
        "aciliyet": "kritik"
    },
    "Tomato__Tomato_mosaic_virus": {
        "neden": "Tomato Mosaic Virus (ToMV)",
        "belirti": "Yapraklarda mozaik/alacalanma, bodurlaşma, meyve küçülmesi",
        "yayilma": "Mekanik temas, kontamine tohumlar, böcekler",
        "ilaclar": ["Doğrudan antiviral yok — önlem kritik"],
        "onlem": "Hasta bitkileri hemen uzaklaştırın, tohumu dezenfekte edin, elleri sık yıkayın",
        "aciliyet": "kritik"
    },
    "Pepper__bell___healthy": {"neden": None},
    "Potato___healthy": {"neden": None},
    "Tomato_healthy": {"neden": None},
}

DISEASE_TR = {
    "Pepper__bell___Bacterial_spot": "Biber - Bakteriyel Leke",
    "Pepper__bell___healthy": "Biber - Sağlıklı",
    "Potato___Early_blight": "Patates - Erken Yanıklık",
    "Potato___Late_blight": "Patates - Geç Yanıklık",
    "Potato___healthy": "Patates - Sağlıklı",
    "Tomato_Bacterial_spot": "Domates - Bakteriyel Leke",
    "Tomato_Early_blight": "Domates - Erken Yanıklık",
    "Tomato_Late_blight": "Domates - Geç Yanıklık",
    "Tomato_Leaf_Mold": "Domates - Yaprak Küfü",
    "Tomato_Septoria_leaf_spot": "Domates - Septoria Yaprak Lekesi",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Domates - Kırmızı Örümcek",
    "Tomato__Target_Spot": "Domates - Hedef Leke",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Domates - Sarı Yaprak Kıvrılma Virüsü",
    "Tomato__Tomato_mosaic_virus": "Domates - Mozaik Virüsü",
    "Tomato_healthy": "Domates - Sağlıklı",
}

# ---- Model ----
@st.cache_resource
def load_model():
    with open("class_names.json") as f:
        class_names = json.load(f)
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model, class_names

model, class_names = load_model()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    tensor = tf(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, 1)
        conf, pred = prob.max(1)
    name = class_names[pred.item()]
    return name, DISEASE_TR.get(name, name), conf.item()

def severity_analysis(image):
    img_array = np.array(image)
    cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([95, 255, 255]))
    d1 = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([25, 255, 200]))
    d2 = cv2.inRange(hsv, np.array([95, 30, 30]), np.array([180, 255, 200]))
    disease_mask = cv2.bitwise_or(d1, d2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)
    leaf = cv2.bitwise_or(green, disease_mask)
    severity = cv2.countNonZero(disease_mask) / max(cv2.countNonZero(leaf), 1) * 100
    heatmap = cv2.applyColorMap(disease_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv_img, 0.6, heatmap, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return severity, disease_mask, overlay_rgb

def get_severity_info(pct):
    if pct < 20:
        return "Hafif", "hafif", "Takip et — henüz müdahale gerekmez"
    elif pct < 50:
        return "Orta", "orta", "Erken dönem ilaçlama önerilir"
    else:
        return "Kritik", "kritik", "Acil müdahale gerekli — komşu bitkileri koru"

def compute_grad_cam(image):
    tensor = tf(image).unsqueeze(0).requires_grad_(True)
    activations, gradients = [], []
    def fwd_hook(m, i, o): activations.append(o.detach())
    def bwd_hook(m, i, o): gradients.append(o[0].detach())
    h1 = model.features[-1].register_forward_hook(fwd_hook)
    h2 = model.features[-1].register_full_backward_hook(bwd_hook)
    out = model(tensor)
    pred = out.argmax(1).item()
    model.zero_grad()
    out[0, pred].backward()
    act = activations[0].squeeze()
    grad = gradients[0].squeeze()
    weights = grad.mean(dim=[1, 2])
    cam = (weights[:, None, None] * act).sum(0).cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (224, 224))
    h1.remove(); h2.remove()
    orig = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * orig + 0.4 * heatmap).astype(np.uint8)
    return cam, overlay

def disease_info_card(disease_key):
    info = DISEASE_INFO.get(disease_key, {})
    if not info or info.get("neden") is None:
        st.markdown("""<div class="disease-card">
            <h3>✅ Sağlıklı Bitki</h3>
            <p style="color:#86efac">Bu yaprakta herhangi bir hastalık belirtisi tespit edilmedi. Düzenli takibe devam edin.</p>
        </div>""", unsafe_allow_html=True)
        return

    aciliyet = info.get("aciliyet", "orta")
    acil_renk = {"hafif": "#22c55e", "orta": "#eab308", "kritik": "#ef4444"}.get(aciliyet, "#eab308")

    ilac_tags = "".join([f'<span class="drug-tag">💊 {i}</span>' for i in info.get("ilaclar", [])])

    st.markdown(f"""
    <div class="disease-card">
        <h3>🔬 Hastalık Bilgi Kartı</h3>
        <div class="info-row">
            <div class="info-chip">
                <div class="info-chip-label">Etken</div>
                <div class="info-chip-val">{info.get('neden', '-')}</div>
            </div>
            <div class="info-chip">
                <div class="info-chip-label">Yayılma</div>
                <div class="info-chip-val">{info.get('yayilma', '-')}</div>
            </div>
        </div>
        <div class="info-chip" style="margin-bottom:0.8rem">
            <div class="info-chip-label">Belirtiler</div>
            <div class="info-chip-val">{info.get('belirti', '-')}</div>
        </div>
        <div class="info-chip-label" style="margin-bottom:6px">Önerilen İlaçlar</div>
        <div style="margin-bottom:0.8rem">{ilac_tags}</div>
        <div class="tip-box">💡 <strong>Önlem:</strong> {info.get('onlem', '-')}</div>
    </div>
    """, unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">🍃 Leaf<span>Guard</span></div>
    <div class="hero-sub">Yaprak Hastalığı Tespiti &nbsp;·&nbsp; Şiddet Analizi &nbsp;·&nbsp; Progresyon Tahmini</div>
</div>
<div class="pill-nav">
    <span class="pill">EfficientNet-B4</span>
    <span class="pill">HSV Analizi</span>
    <span class="pill">Grad-CAM XAI</span>
    <span class="pill">Polinomyal Regresyon</span>
    <span class="pill">15 Hastalık Sınıfı</span>
</div>
""", unsafe_allow_html=True)

# ========== TABS ==========
tab1, tab2 = st.tabs(["🔬  Tekli Analiz", "📈  Progresyon Tahmini"])

with tab1:
    st.markdown('<div class="sec-label">Görüntü Yükle</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Yaprak fotoğrafı seçin (JPG / PNG)", type=["jpg", "jpeg", "png"], key="single")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        disease_key, disease_tr, confidence = predict(image)
        severity, mask, overlay = severity_analysis(image)
        level, level_key, action = get_severity_info(severity)
        cam, cam_overlay = compute_grad_cam(image)

        st.markdown("---")
        st.markdown('<div class="sec-label">Analiz Sonuçları</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Tespit Edilen Hastalık</div>
                <div class="metric-val" style="font-size:1rem">{disease_tr}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            col = "green" if confidence > 0.9 else "yellow" if confidence > 0.7 else "red"
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Model Güveni</div>
                <div class="metric-val {col}">%{confidence*100:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-label">Şiddet</div>
                <div class="metric-val">%{severity:.1f}</div>
                <span class="badge badge-{level_key}">{level}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="action-strip {level_key}">📋 <strong>Öneri:</strong> {action}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-label" style="margin-top:1.5rem">Görsel Analiz</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.image(image,       caption="Orijinal",         use_container_width=True)
        c2.image(mask,        caption="Hastalık Maskesi",  use_container_width=True)
        c3.image(overlay,     caption="Şiddet Isı Haritası", use_container_width=True)
        c4.image(cam_overlay, caption="Grad-CAM (XAI)",   use_container_width=True)

        # Hastalık bilgi kartı
        st.markdown('<div class="sec-label" style="margin-top:1.5rem">Hastalık Bilgi Kartı</div>', unsafe_allow_html=True)
        disease_info_card(disease_key)

with tab2:
    st.markdown('<div class="sec-label">Çok Zamanlı Progresyon Analizi</div>', unsafe_allow_html=True)
    st.markdown("Aynı bitkiye ait **farklı günlerin** görüntülerini yükleyerek hastalığın ilerleyişini tahmin edin.")

    num_days = st.slider("Kaç gün verisi var?", 2, 7, 3)
    days_data = []
    cols = st.columns(num_days)

    for i in range(num_days):
        with cols[i]:
            day = st.number_input(f"Gün {i+1}", min_value=1, max_value=90, value=(i+1)*3, key=f"day_{i}")
            img_file = st.file_uploader(f"Gün {int(day)}", type=["jpg","jpeg","png"], key=f"img_{i}")
            if img_file:
                img = Image.open(img_file).convert("RGB")
                st.image(img, use_container_width=True)
                sev, _, _ = severity_analysis(img)
                level, level_key, _ = get_severity_info(sev)
                st.markdown(f'<span class="badge badge-{level_key}">%{sev:.1f} — {level}</span>', unsafe_allow_html=True)
                days_data.append((day, sev))

    if len(days_data) >= 2:
        days_arr = np.array([d[0] for d in days_data]).reshape(-1, 1)
        sevs_arr = np.array([d[1] for d in days_data])
        poly = PolynomialFeatures(degree=2)
        reg = LinearRegression().fit(poly.fit_transform(days_arr), sevs_arr)
        future = np.arange(1, 30).reshape(-1, 1)
        preds = np.clip(reg.predict(poly.transform(future)), 0, 100)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#060d06')
        ax.set_facecolor('#0a180a')
        ax.scatter(days_arr, sevs_arr, color='#22c55e', s=120, zorder=5, label='Gerçek Ölçüm', edgecolors='#f0fdf4', linewidths=1)
        ax.plot(future, preds, color='#4ade80', linewidth=2, linestyle='--', label='Tahmin Eğrisi')
        ax.axhline(y=50, color='#ef4444', linestyle=':', alpha=0.7, label='Kritik Eşik (%50)')
        critical = future[preds >= 50]
        if len(critical) > 0:
            cd = critical[0][0]
            ax.axvline(x=cd, color='#ef4444', linestyle=':', alpha=0.5)
            ax.text(cd + 0.5, 52, f'{int(cd)}. gün', color='#ef4444', fontsize=11, fontweight='bold')
            st.error(f"⚠️ Kritik eşiğe tahmini ulaşma: **{int(cd)}. gün** — Acil müdahale planı yapın!")
        else:
            st.success("✅ 30 gün içinde kritik eşiğe ulaşmıyor — düzenli takip yeterli.")

        ax.set_xlabel('Gün', color='#4ade80', fontsize=11)
        ax.set_ylabel('Etkilenen Alan (%)', color='#4ade80', fontsize=11)
        ax.set_title('Hastalık Progresyon Tahmini', color='#f0fdf4', fontsize=13, fontweight='bold', pad=15)
        ax.tick_params(colors='#4ade80')
        ax.legend(facecolor='#0a180a', edgecolor='#1a3a1a', labelcolor='#d1fae5')
        ax.grid(True, alpha=0.1, color='#22c55e')
        for spine in ax.spines.values():
            spine.set_color('#1a3a1a')
        st.pyplot(fig)

st.markdown("""<div class="footer">
    LEAFGUARD v2.0 &nbsp;·&nbsp; EfficientNet-B4 + HSV + Grad-CAM + Regresyon &nbsp;·&nbsp;
    Fırat Üniversitesi — Yazılım Mühendisliği
</div>""", unsafe_allow_html=True)