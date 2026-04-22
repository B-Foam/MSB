import io
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURAÇÃO DA PÁGINA E ESTADO
# ============================================================
st.set_page_config(page_title="Análise Granulométrica Ultrarrapida", layout="wide")

# CSS para garantir que a imagem não transborde e a interface fique limpa
st.markdown("""
<style>
    .stImage > img {
        max-width: 100%;
        height: auto;
    }
</style>
""", unsafe_allow_stdio=True)

def garantir_estado():
    # Usamos o hash do arquivo como chave para o estado
    if "data_historico" not in st.session_state:
        st.session_state.data_historico = {}


# ============================================================
# MÓDULO 1: UTILITÁRIOS DE IMAGEM (RÁPIDOS)
# ============================================================
def carregar_imagem_local(uploaded_file) -> np.ndarray:
    """Carrega imagem do uploader local e converte para BGR OpenCV."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Não foi possível decodificar a imagem.")
    return img_bgr

def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    """Converte imagem OpenCV (BGR ou Grayscale) para PIL para o Streamlit."""
    if len(img_cv.shape) == 2: # Grayscale
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ============================================================
# MÓDULO 2: DETECÇÃO DA BARRA DE ESCALA
# ============================================================
def detectar_barra_escala_px(img_bgr: np.ndarray):
    """Detecta automaticamente a barra de escala de 1mm na região inferior esquerda."""
    h, w = img_bgr.shape[:2]
    # Região de busca inferior esquerda
    x0, x1 = 0, int(w * 0.35)
    y0, y1 = int(h * 0.78), h
    crop = img_bgr[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Threshold para encontrar elementos escuros
    _, th = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor = None
    melhor_area = 0
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / max(hh, 1)
        area = ww * hh
        # Critérios para barra horizontal longa
        if ww > 60 and hh < 25 and aspect > 4.5 and area > melhor_area:
            melhor_area = area
            melhor = (x, y, ww, hh)

    if melhor is None:
        return None, None
    x, y, ww, hh = melhor
    barra_info = {"x": x0 + x, "y": y0 + y, "w": ww, "h": hh}
    return float(ww), barra_info


# ============================================================
# MÓDULO 3: GESTÃO DA ROI (REGION OF INTEREST)
# ============================================================
def criar_roi_padrao(shape: Tuple[int, int, int]):
    h, w = shape[:2]
    return {"cx": w // 2, "cy": h // 2, "r": int(min(h, w) * 0.38)}

def criar_mascara_roi(shape, roi_info: Dict[str, int], barra_info=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 255, -1)
    # Remove cabeçalho superior e região da barra
    mask[:int(h * 0.045), :] = 0
    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        mask[max(0, y - 50):min(h, y + hh + 20), max(0, x - 20):min(w, x + ww + 90)] = 0
    return mask


# ============================================================
# MÓDULO 4: NOVO PIPELINE DE DETECÇÃO ULTRARRAPIDO (WATERSHED)
# ============================================================
def preprocessar_para_watershed(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 1. Filtro Bilateral Pesado: Suaviza o interior mantendo bordas
    denoised = cv2.bilateralFilter(gray, 15, 80, 80)
    # 2. Threshold Adaptativo Gaussiano (Invertido)
    img_roi = cv2.bitwise_and(denoised, denoised, mask=mask_roi)
    thresh = cv2.adaptiveThreshold(img_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    # 3. Limpeza morfológica (Opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return gray, opening

def detectar_bolhas_watershed(img_gray, opening, mask_roi, roi_info):
    # 4. Transformada de distância (separa bolhas conectadas)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # 5. Threshold nos marcadores (Foreground Seguro)
    _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
    # 6. Watershed
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    cv2.circle(markers, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 0, 2)
    markers[opening == 0] = 0
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(img_bgr, markers)
    # 7. Extração de Métricas
    bolhas_detectadas = []
    for label in np.unique(labels):
        if label <= 1: continue
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        mask[labels == label] = 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) > 20: 
                (x, y), r = cv2.minEnclosingCircle(cnt)
                bolhas_detectadas.append({"x": x, "y": y, "r": r})
    return bolhas_detectadas


# ============================================================
# MÓDULO 5: DESENHOS E VISUALIZAÇÃO
# ============================================================
def desenhar_imagem_roi(img_bgr: np.ndarray, roi_info: Dict[str, int], barra_info=None) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (roi_info["cx"], roi_info["cy"]), roi_info["r"], 255, -1)
    overlay[mask == 0] = (25, 25, 25) # Escurece o fundo
    out = cv2.addWeighted(out, 0.62, overlay, 0.38, 0)
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), 2)
    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        cv2.line(out, (x, y + hh // 2), (x + ww, y + hh // 2), (0, 255, 0), 2)
    return out

def desenhar_bolhas_coloridas(shape, roi_info: Dict[str, int], bolhas: List[Dict], barra_info=None) -> np.ndarray:
    h, w = shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # fundo preto + área útil branca
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), -1)
    rng = np.random.default_rng(123)
    for i, b in enumerate(bolhas, start=1):
        x, y, r = int(round(b["x"])), int(round(b["y"])), int(round(b["r"]))
        color = tuple(int(v) for v in rng.integers(30, 256, size=3))
        cv2.circle(out, (x, y), r, color, -1) # preenchimento colorido
        thickness = max(1, int(round(r * 0.10)))
        cv2.circle(out, (x, y), r, (0, 0, 0), thickness) # contorno preto
        if r >= 11:
            cv2.putText(out, str(i), (x - int(r * 0.35), y + int(r * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    # Contorno externo e Barra
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), 2)
    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        cv2.putText(out, "1.0 mm", (x + ww + 10, y + hh), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return out


# ============================================================
# INTERFACE PRINCIPAL (STREAMLIT ULTRARRAPIDA)
# ============================================================
def main():
    st.title("🔬 Análise Granulométrica Ultrarrapida")
    st.markdown("---")
    
    garantir_estado()
    
    # Módulo de Upload Local (Resolve a lentidão do Supabase)
    uploaded_file = st.file_uploader("Selecione a imagem de espuma para análise", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        st.info("Aguardando upload da imagem...")
        return

    # Gera uma chave única baseada no nome do arquivo para o cache de estado
    file_id = uploaded_file.name
    if file_id not in st.session_state.data_historico:
        st.session_state.data_historico[file_id] = {"roi_info": None, "bolhas": None}
    
    # Carregamento local é instantâneo
    with st.spinner("Carregando imagem local..."):
        img_bgr = carregar_imagem_local(uploaded_file)
        h, w = img_bgr.shape[:2]
        px_per_mm, barra_info = detectar_barra_escala_px(img_bgr)
    
    # Recupera ou cria ROI padrão
    if st.session_state.data_historico[file_id]["roi_info"] is None:
        st.session_state.data_historico[file_id]["roi_info"] = criar_roi_padrao(img_bgr.shape)
    roi_info = st.session_state.data_historico[file_id]["roi_info"]
    
    col_img, col_ctrl = st.columns([3, 1])
    
    with col_ctrl:
        st.subheader("Configurações")
        st.write(f"Resolução: {w}x{h} px")
        
        # Ajuste da ROI
        with st.expander("Ajuste da Área Útil (ROI)", expanded=True):
            roi_info["cx"] = st.number_input("Centro X", 0, w, int(roi_info["cx"]), step=10, key=f"cx_{file_id}")
            roi_info["cy"] = st.number_input("Centro Y", 0, h, int(roi_info["cy"]), step=10, key=f"cy_{file_id}")
            roi_info["r"] = st.number_input("Raio", 50, min(h, w)//2, int(roi_info["r"]), step=10, key=f"r_{file_id}")
            # Salva ROI ajustada
            st.session_state.data_historico[file_id]["roi_info"] = roi_info
        
        st.markdown("---")
        processar = st.button("🚀 Detectar Bolhas", type="primary", use_container_width=True, key=f"btn_{file_id}")

    with col_img:
        st.markdown(f"#### Arquivo: `{file_id}`")
        tab1, tab2 = st.tabs(["🖼️ Área Útil & Calibração", "📊 Resultado (Instância)"])
        
        # Gera máscara ROI atualizada
        mask_roi = criar_mascara_roi(img_bgr.shape, roi_info, barra_info)
        
        with tab1:
            # Imagem 1 — área útil + calibração (carrega instantaneamente)
            img_roi = desenhar_imagem_roi(img_bgr, roi_info, barra_info=barra_info)
            st.image(cv_to_pil(img_roi), use_container_width=True, caption="Visualização da ROI e Barra de Calibração")
            
            if px_per_mm: st.success(f"Calibração Automática: 1,0 mm = {px_per_mm:.2f} px")
            else: st.warning("Barra de calibração não detectada. Medições físicas indisponíveis.")

        with tab2:
            # Processamento de detecção ultrarrapido
            if processar:
                # O spinner de processamento só aparece agora, sobre o local da imagem
                with st.spinner("Executando detecção (Watershed)..."):
                    img_gray, opening = preprocessar_para_watershed(img_bgr, mask_roi)
                    bolhas = detectar_bolhas_watershed(img_gray, opening, mask_roi, roi_info)
                    # Salva resultado no histórico do arquivo
                    st.session_state.data_historico[file_id]["bolhas"] = bolhas
            
            # Recupera bolhas do histórico (se houver) para renderizar a imagem final
            bolhas_finais = st.session_state.data_historico[file_id]["bolhas"]
            
            if bolhas_finais is not None:
                # Geração da imagem final (também rápida)
                img_final = desenhar_bolhas_coloridas(shape=img_bgr.shape, roi_info=roi_info, bolhas=bolhas_finais, barra_info=barra_info)
                st.image(cv_to_pil(img_final), use_container_width=True)
                st.metric("Total de Bolhas", len(bolhas_finais))
            else:
                st.info("Clique em '🚀 Detectar Bolhas' para visualizar o resultado.")

if __name__ == "__main__":
    main()
