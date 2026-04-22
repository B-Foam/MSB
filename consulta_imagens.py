import io
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURAÇÃO DA PÁGINA E ESTADO
# ============================================================
st.set_page_config(page_title="Análise de Bolhas Ultrarrapida", layout="wide")

def garantir_estado():
    if "lista_imagens_consulta" not in st.session_state:
        st.session_state.lista_imagens_consulta = []
    if "roi_consulta" not in st.session_state:
        st.session_state.roi_consulta = {}
    if "bolhas_detectadas" not in st.session_state:
        st.session_state.bolhas_detectadas = {}


# ============================================================
# MÓDULO 1: SIMULAÇÃO DO SUPABASE (PARA TESTE LOCAL)
# Substitua estas funções pelas suas reais que conectam ao Supabase
# ============================================================
def listar_imagens_supabase_mock(path: str):
    """Simula a listagem de imagens do Supabase."""
    # Lista de URLs de imagens públicas para teste
    imagens_teste = [
        {"name": "Espuma_A.jpg", "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/detect_blob.blob.png"}, # Imagem de teste sintética
        {"name": "Espuma_B.jpg", "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Bierpong_Tisch_Freiburg_2013.jpg/1280px-Bierpong_Tisch_Freiburg_2013.jpg"}, # Imagem com bolhas reais
    ]
    return imagens_teste, None

def montar_url_publica_mock(nome_imagem: str):
    """Simula a montagem da URL pública do Supabase."""
    imagens, _ = listar_imagens_supabase_mock("")
    for img in imagens:
        if img["name"] == nome_imagem:
            return img["url"]
    return None


# ============================================================
# MÓDULO 2: UTILITÁRIOS DE IMAGEM E CACHE
# ============================================================
@st.cache_data(show_spinner=False)
def baixar_imagem_bytes(url: str) -> bytes:
    """Baixa a imagem e armazena os bytes em cache."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content

def abrir_imagem(url: str) -> Image.Image:
    """Abre uma imagem PIL a partir de uma URL."""
    data = baixar_imagem_bytes(url)
    return Image.open(io.BytesIO(data)).convert("RGB")

def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    """Converte imagem PIL para BGR OpenCV."""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    """Converte imagem OpenCV (BGR ou Grayscale) para PIL."""
    if len(img_cv.shape) == 2: # Grayscale
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ============================================================
# MÓDULO 3: DETECÇÃO DA BARRA DE ESCALA
# ============================================================
def detectar_barra_escala_px(img_bgr: np.ndarray):
    """Detecta automaticamente a barra de escala de 1mm."""
    img_annot = img_bgr.copy()
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
        return None, img_annot, None

    x, y, ww, hh = melhor
    gx, gy = x0 + x, y0 + y # Coordenadas globais

    # Anotação na imagem
    cv2.rectangle(img_annot, (gx, gy), (gx + ww, gy + hh), (0, 255, 0), 2)
    cv2.putText(img_annot, f"1.0 mm = {ww} px", (gx, max(20, gy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    
    barra_info = {"x": gx, "y": gy, "w": ww, "h": hh}
    return float(ww), img_annot, barra_info


# ============================================================
# MÓDULO 4: GESTÃO DA ROI (REGION OF INTEREST)
# ============================================================
def criar_roi_padrao(shape: Tuple[int, int, int]):
    """Cria uma ROI circular padrão no centro da imagem."""
    h, w = shape[:2]
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.38)
    return {"cx": cx, "cy": cy, "r": r}

def criar_mascara_roi(shape, roi_info: Dict[str, int], barra_info=None):
    """Cria uma máscara binária da área útil (ROI)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Área útil circular (branca)
    cv2.circle(mask, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 255, -1)

    # Remove cabeçalho superior (preto)
    topo = int(h * 0.045)
    mask[:topo, :] = 0

    # Remove região da barra de escala (preto)
    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        x_ini, y_ini = max(0, x - 20), max(0, y - 50)
        x_fim, y_fim = min(w, x + ww + 90), min(h, y + hh + 20)
        mask[y_ini:y_fim, x_ini:x_fim] = 0

    return mask


# ============================================================
# MÓDULO 5: NOVO PIPELINE DE DETECÇÃO ULTRARRAPIDO (WATERSHED)
# ============================================================
def preprocessar_para_watershed(img_bgr: np.ndarray, mask_roi: np.ndarray):
    """Pré-processamento agressivo para limpar ruído de textura."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Filtro Bilateral Pesado: Suaviza o interior mantendo bordas
    # Essencial para remover a textura ruidosa da espuma
    denoised = cv2.bilateralFilter(gray, 15, 80, 80)
    
    # 2. Threshold Adaptativo Gaussiano (Invertido)
    # Isola as paredes das bolhas dentro da ROI
    img_roi = cv2.bitwise_and(denoised, denoised, mask=mask_roi)
    thresh = cv2.adaptiveThreshold(img_roi, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    # 3. Limpeza morfológica (Opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return gray, opening

def detectar_bolhas_watershed(img_gray, opening, mask_roi, roi_info):
    """Segmentação por Watershed para separar e contar bolhas."""
    # 4. Transformada de distância: Essencial para separar bolhas grudadas
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # 5. Threshold nos marcadores (Foreground Seguro)
    # Define o centro de cada bolha. Aumente o fator (0.35) para separar mais.
    fator_separacao = 0.35
    _, sure_fg = cv2.threshold(dist_transform, fator_separacao * dist_transform.max(), 255, 0)
    
    # 6. Preparação dos Marcadores para Watershed
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 # Garante que o fundo não seja 0
    
    # Fronteira de ROI: Impede que o Watershed 'vaze' para fora da área útil
    cv2.circle(markers, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 0, 2)
    
    # Define o fundo (área fora da ROI ou paredes) como região desconhecida
    markers[opening == 0] = 0
    
    # Aplica o algoritmo Watershed
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(img_bgr, markers)
    
    # 7. Extração de Métricas dos Objetos Segmentados
    bolhas_detectadas = []
    # Itera sobre cada label único (ignora label 1 que é o fundo)
    for label in np.unique(labels):
        if label <= 1: continue
        
        # Cria máscara para a bolha atual
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        mask[labels == label] = 255
        
        # Encontra o contorno exato
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            # Filtro de área mínima para ignorar artefatos
            if cv2.contourArea(cnt) > 20: 
                # Encontra o círculo mínimo que engloba o contorno
                (x, y), r = cv2.minEnclosingCircle(cnt)
                bolhas_detectadas.append({"x": x, "y": y, "r": r})
                
    return bolhas_detectadas


# ============================================================
# MÓDULO 6: DESENHOS E VISUALIZAÇÃO
# ============================================================
def desenhar_imagem_roi(img_bgr: np.ndarray, roi_info: Dict[str, int], barra_info=None) -> np.ndarray:
    """Desenha o sombreamento da ROI e a calibração."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    # Área útil circular
    cv2.circle(mask, (roi_info["cx"], roi_info["cy"]), roi_info["r"], 255, -1)
    # Escurece o fundo
    overlay[mask == 0] = (25, 25, 25)
    out = cv2.addWeighted(out, 0.62, overlay, 0.38, 0)
    # Contorno branco da ROI
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), 2)
    return out

def desenhar_bolhas_coloridas(shape, roi_info: Dict[str, int], bolhas: List[Dict], barra_info=None) -> np.ndarray:
    """Desenha as bolhas segmentadas com cores aleatórias (estilo Instância)."""
    h, w = shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # Área útil branca (fundo)
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), -1)
    
    rng = np.random.default_rng(123) # Seed fixa para cores consistentes

    for i, b in enumerate(bolhas, start=1):
        x, y, r = int(round(b["x"])), int(round(b["y"])), int(round(b["r"]))
        # Cor aleatória vibrante
        color = tuple(int(v) for v in rng.integers(30, 256, size=3))
        
        # Preenchimento colorido
        cv2.circle(out, (x, y), r, color, -1)
        # Contorno preto fino
        thickness = max(1, int(round(r * 0.10)))
        cv2.circle(out, (x, y), r, (0, 0, 0), thickness)
        
        # ID para bolhas maiores
        if r >= 11:
            cv2.putText(out, str(i), (x - int(r * 0.35), y + int(r * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            
    # Contorno externo da ROI
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), 2)
    
    # Barra de escala (legenda)
    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        cv2.putText(out, "1.0 mm", (x + ww + 10, y + hh), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        
    return out


# ============================================================
# MÓDULO 7: RENDERIZAÇÃO DA INTERFACE (STREAMLIT)
# ============================================================
def main():
    st.title("🔬 Análise de Bolhas Ultrarrapida (Watershed)")
    st.markdown("---")
    
    garantir_estado()
    
    # Define quais funções do Supabase usar (mudar para reais em produção)
    fn_listar = listar_imagens_supabase_mock
    fn_url = montar_url_publica_mock

    with st.sidebar:
        st.header("Configurações")
        if st.button("🔄 Atualizar lista de imagens"):
            with st.spinner("Conectando ao Supabase..."):
                imagens, erro = fn_listar("")
                if erro:
                    st.error(f"Erro ao listar imagens: {erro}")
                    st.stop()
                st.session_state.lista_imagens_consulta = [img["name"] for img in imagens]
                st.success("Lista atualizada!")
        
        lista = st.session_state.lista_imagens_consulta
        if not lista:
            st.info("Clique em 'Atualizar lista' para carregar.")
            st.stop()
        
        imagem_escolhida = st.selectbox("Selecione a imagem", lista)

    # Carregamento e Calibração
    with st.spinner("Carregando imagem..."):
        url = fn_url(imagem_escolhida)
        img_pil = abrir_imagem(url)
        img_bgr = pil_to_cv(img_pil)
        px_per_mm, _, barra_info = detectar_barra_escala_px(img_bgr)
    
    # Definição/Ajuste da ROI por imagem
    if imagem_escolhida not in st.session_state.roi_consulta:
        st.session_state.roi_consulta[imagem_escolhida] = criar_roi_padrao(img_bgr.shape)
    roi_info = st.session_state.roi_consulta[imagem_escolhida]
    
    col_img, col_ctrl = st.columns([3, 1])
    
    with col_ctrl:
        st.subheader("Ajuste da Área Útil")
        roi_info["cx"] = st.number_input("Centro X", 0, img_bgr.shape[1], int(roi_info["cx"]), key=f"cx_{imagem_escolhida}")
        roi_info["cy"] = st.number_input("Centro Y", 0, img_bgr.shape[0], int(roi_info["cy"]), key=f"cy_{imagem_escolhida}")
        roi_info["r"] = st.number_input("Raio", 50, min(img_bgr.shape[:2])//2, int(roi_info["r"]), key=f"r_{imagem_escolhida}")
        
        # Atualiza estado da ROI
        st.session_state.roi_consulta[imagem_escolhida] = roi_info
        mask_roi = criar_mascara_roi(img_bgr.shape, roi_info, barra_info)
        
        st.markdown("---")
        st.subheader("Processamento")
        processar = st.button("🚀 Detectar Bolhas", type="primary", use_container_width=True)

    with col_img:
        st.markdown(f"### Visualização: {imagem_escolhida}")
        
        # Cria tabs para Imagem Original e Resultado
        tab1, tab2 = st.tabs(["🖼️ Área Útil & Calibração", "📊 Resultado da Detecção"])
        
        with tab1:
            # Imagem 1 — área útil + calibração
            img_roi = desenhar_imagem_roi(img_bgr, roi_info, barra_info=barra_info)
            st.image(cv_to_pil(img_roi), use_container_width=True, caption="Área útil circular (ROI)")
            
            if px_per_mm:
                st.success(f"Calibração Automática: 1,0 mm = {px_per_mm:.2f} px")
            else:
                st.warning("Barra de calibração não detectada. Medições físicas estarão indisponíveis.")

        with tab2:
            # Lógica de processamento quando o botão é clicado
            if processar:
                with st.spinner("Executando detecção ultrarrapida (Watershed)..."):
                    # Pipeline Otimizado
                    img_gray, opening = preprocessar_para_watershed(img_bgr, mask_roi)
                    bolhas = detectar_bolhas_watershed(img_gray, opening, mask_roi, roi_info)
                    
                    # Salva resultado no estado
                    st.session_state.bolhas_detectadas[imagem_escolhida] = bolhas
                    
                    # Geração da imagem final
                    img_final = desenhar_bolhas_coloridas(shape=img_bgr.shape, roi_info=roi_info, bolhas=bolhas, barra_info=barra_info)
                    
                    st.image(cv_to_pil(img_final), use_container_width=True, caption=f"Resultado: {len(bolhas)} bolhas detectadas.")
                    st.metric("Bolhas Detectadas", len(bolhas))
            
            # Se já houver detecção prévia no estado, mostra
            elif imagem_escolhida in st.session_state.bolhas_detectadas:
                bolhas = st.session_state.bolhas_detectadas[imagem_escolhida]
                img_final = desenhar_bolhas_coloridas(shape=img_bgr.shape, roi_info=roi_info, bolhas=bolhas, barra_info=barra_info)
                st.image(cv_to_pil(img_final), use_container_width=True)
                st.info(f"Mostrando detecção anterior. Clique em 'Detectar Bolhas' para reprocessar.")
            
            else:
                st.info("Ajuste a ROI na barra lateral e clique em 'Detectar Bolhas'.")

if __name__ == "__main__":
    main()
