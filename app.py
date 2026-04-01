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

# ---- Sayfa Ayarı ----
st.set_page_config(
    page_title="LeafGuard - Yaprak Hastalığı Analiz Sistemi",
    page_icon="🍃",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { background-color: #0a0a0a; }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #22c55e, #16a34a, #15803d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-label {
        font-family: 'Inter', sans-serif;
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .metric-value-green { color: #22c55e; }
    .metric-value-yellow { color: #eab308; }
    .metric-value-red { color: #ef4444; }

    .severity-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        font-family: 'Inter', sans-serif;
    }
    .badge-hafif { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid #22c55e; }
    .badge-orta { background: rgba(234,179,8,0.15); color: #eab308; border: 1px solid #eab308; }
    .badge-kritik { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid #ef4444; }

    .action-box {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-left: 4px solid;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    .action-hafif { border-color: #22c55e; }
    .action-orta { border-color: #eab308; }
    .action-kritik { border-color: #ef4444; }

    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #22c55e;
        display: inline-block;
    }

    .upload-area {
        background: linear-gradient(145deg, #1a1a2e, #0d1117);
        border: 2px dashed #2d3748;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.3s;
    }
    .upload-area:hover { border-color: #22c55e; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111827;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }

    div[data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2d3748;
    }

    .footer {
        text-align: center;
        color: #4b5563;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #1f2937;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<div class="hero-title">🍃 LeafGuard</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Yaprak Hastalığı Tespiti • Şiddet Haritalaması • Progresyon Tahmini</div>', unsafe_allow_html=True)

# ---- Model Yükleme ----
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

def predict(image):
    tensor = tf(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        prob = torch.softmax(out, 1)
        conf, pred = prob.max(1)
    name = class_names[pred.item()]
    return DISEASE_TR.get(name, name), conf.item()

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
        return "Hafif", "hafif", "Takip et, henüz müdahale gerekmez"
    elif pct < 50:
        return "Orta", "orta", "Erken dönem ilaçlama önerilir"
    else:
        return "Kritik", "kritik", "Acil müdahale gerekli, komşu bitkileri koru"

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

# ========== TABS ==========
tab1, tab2 = st.tabs(["🔬 Tekli Analiz", "📈 Progresyon Tahmini"])

with tab1:
    uploaded = st.file_uploader("Yaprak görüntüsü yükleyin", type=["jpg", "jpeg", "png"], key="single")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        disease, confidence = predict(image)
        severity, mask, overlay = severity_analysis(image)
        level, level_key, action = get_severity_info(severity)
        cam, cam_overlay = compute_grad_cam(image)

        # Metrikler
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Tespit Edilen Hastalık</div>
                <div class="metric-value">{disease}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            color_class = "green" if confidence > 0.9 else "yellow" if confidence > 0.7 else "red"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Model Güveni</div>
                <div class="metric-value metric-value-{color_class}">%{confidence*100:.1f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Şiddet Seviyesi</div>
                <div class="metric-value">%{severity:.1f}</div>
                <span class="severity-badge badge-{level_key}">{level}</span>
            </div>""", unsafe_allow_html=True)

        # Öneri
        st.markdown(f"""<div class="action-box action-{level_key}">
            <strong>📋 Öneri:</strong> {action}
        </div>""", unsafe_allow_html=True)

        # Görseller
        st.markdown('<div class="section-header">📊 Görsel Analiz</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.image(image, caption="Orijinal Görüntü", use_container_width=True)
        c2.image(mask, caption="Hastalık Maskesi", use_container_width=True)
        c3.image(overlay, caption="Şiddet Isı Haritası", use_container_width=True)
        c4.image(cam_overlay, caption="Grad-CAM (XAI)", use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">📈 Çok Zamanlı Progresyon Analizi</div>', unsafe_allow_html=True)
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
                st.markdown(f'<span class="severity-badge badge-{level_key}">%{sev:.1f} - {level}</span>', unsafe_allow_html=True)
                days_data.append((day, sev))

    if len(days_data) >= 2:
        days_arr = np.array([d[0] for d in days_data]).reshape(-1, 1)
        sevs_arr = np.array([d[1] for d in days_data])
        poly = PolynomialFeatures(degree=2)
        reg = LinearRegression().fit(poly.fit_transform(days_arr), sevs_arr)
        future = np.arange(1, 30).reshape(-1, 1)
        preds = np.clip(reg.predict(poly.transform(future)), 0, 100)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#111827')
        ax.scatter(days_arr, sevs_arr, color='#22c55e', s=120, zorder=5, label='Gerçek Ölçüm', edgecolors='white')
        ax.plot(future, preds, color='#3b82f6', linewidth=2, linestyle='--', label='Tahmin Eğrisi')
        ax.axhline(y=50, color='#ef4444', linestyle=':', alpha=0.7, label='Kritik Eşik (%50)')
        critical = future[preds >= 50]
        if len(critical) > 0:
            cd = critical[0][0]
            ax.axvline(x=cd, color='#ef4444', linestyle=':', alpha=0.5)
            ax.text(cd + 0.5, 52, f'{int(cd)}. gün', color='#ef4444', fontsize=12, fontweight='bold')
            st.error(f"⚠️ Kritik eşiğe tahmini ulaşma: **{int(cd)}. gün** — Acil müdahale planı yapın!")
        else:
            st.success("✅ 30 gün içinde kritik eşiğe ulaşmıyor — düzenli takip yeterli.")
        ax.set_xlabel('Gün', color='white', fontsize=12)
        ax.set_ylabel('Etkilenen Alan (%)', color='white', fontsize=12)
        ax.set_title('Hastalık Progresyon Tahmini', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1f2937', edgecolor='#374151', labelcolor='white')
        ax.grid(True, alpha=0.15, color='white')
        ax.spines['bottom'].set_color('#374151')
        ax.spines['left'].set_color('#374151')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

# Footer
st.markdown("""<div class="footer">
    LeafGuard v1.0 • EfficientNet-B4 + HSV Analizi + Grad-CAM <br>
    Fırat Üniversitesi — Yazılım Mühendisliği
</div>""", unsafe_allow_html=True)