import io
import math
from typing import Dict, List, Tuple, Any, Optional

import requests
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================
def baixar_imagem(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    if len(img_cv.shape) == 2:
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ============================================================
# ESTADO E ROI
# ============================================================
def garantir_estado(session_state):
    if "lista_imagens_consulta" not in session_state:
        session_state.lista_imagens_consulta = []
    if "circles_by_image" not in session_state:
        session_state.circles_by_image = {}
    if "roi_by_image" not in session_state:
        session_state.roi_by_image = {}

def obter_circulos(session_state, image_name: str) -> List[Dict[str, Any]]:
    garantir_estado(session_state)
    return session_state.circles_by_image.get(image_name, [])

def salvar_circulos(session_state, image_name: str, circles: List[Dict[str, Any]]):
    garantir_estado(session_state)
    session_state.circles_by_image[image_name] = circles

def obter_roi(session_state, image_name: str) -> Optional[Dict[str, Any]]:
    garantir_estado(session_state)
    return session_state.roi_by_image.get(image_name)

def salvar_roi(session_state, image_name: str, roi_info: Dict[str, Any]):
    garantir_estado(session_state)
    session_state.roi_by_image[image_name] = roi_info

def criar_roi_circular_padrao(shape, raio_frac=0.43):
    h, w = shape[:2]
    cx = w // 2
    cy = h // 2
    r = int(min(h, w) * raio_frac)
    return {"cx": cx, "cy": cy, "r": r}

def criar_mascara_roi(shape, roi_info: Dict[str, Any], barra_info=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 255, -1)
    mask[0:int(h * 0.035), :] = 0
    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        mask[max(0, y - 60):min(h, y + hh + 20), max(0, x - 25):min(w, x + ww + 90)] = 0
    return mask

def ponto_dentro_roi(x: float, y: float, roi_info: Dict[str, Any], folga: float = 0.0) -> bool:
    dx = x - roi_info["cx"]
    dy = y - roi_info["cy"]
    return (dx * dx + dy * dy) <= (roi_info["r"] - folga) ** 2

# ============================================================
# CALIBRAÇÃO DA BARRA (MANTIDA)
# ============================================================
def detectar_barra_escala_px(img_bgr: np.ndarray):
    img_annot = img_bgr.copy()
    h, w = img_bgr.shape[:2]
    crop = img_bgr[int(h * 0.78):h, 0:int(w * 0.35)].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    melhor = None
    melhor_area = 0
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if ww > 60 and hh < 20 and (ww/max(hh,1)) > 5 and area > melhor_area:
            melhor_area = area
            melhor = (x, y, ww, hh)
    if melhor is None: return None, img_annot, None
    x, y, ww, hh = melhor
    gx, gy = x, int(h * 0.78) + y
    cv2.rectangle(img_annot, (gx, gy), (gx + ww, gy + hh), (0, 255, 0), 2)
    cv2.putText(img_annot, f"1.0 mm = {ww} px", (gx, max(20, gy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return float(ww), img_annot, {"x": gx, "y": gy, "w": ww, "h": hh}

# ============================================================
# NOVO PIPELINE DE PROCESSAMENTO E DETECÇÃO (HOUGH MULTICANAL)
# ============================================================
def preprocessar_otimizado(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Limpeza pesada para remover ruído de textura
    # Aumentamos o filtro bilateral para suavizar o interior das bolhas
    bilateral = cv2.bilateralFilter(gray, 15, 80, 80)
    
    # 2. Realce de bordas (Sharpen)
    blur_ref = cv2.GaussianBlur(bilateral, (0, 0), 2.0)
    sharpen = cv2.addWeighted(bilateral, 1.8, blur_ref, -0.8, 0)
    
    # 3. LoG (Laplacian of Gaussian) absoluto para detectar centros
    log_img = cv2.Laplacian(sharpen, cv2.CV_32F, ksize=3)
    log_abs = cv2.normalize(np.abs(log_img), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Aplica ROI nas imagens de processamento
    proc_bilateral = cv2.bitwise_and(bilateral, bilateral, mask=mask_roi)
    proc_sharpen = cv2.bitwise_and(sharpen, sharpen, mask=mask_roi)
    proc_log = cv2.bitwise_and(log_abs, log_abs, mask=mask_roi)
    
    return {
        "gray_vis": gray,
        "bilateral_vis": bilateral,
        "sharpen_vis": sharpen,
        "log_vis": log_abs,
        "bilateral_proc": proc_bilateral,
        "sharpen_proc": proc_sharpen,
        "log_proc": proc_log
    }

def detectar_hough_multiescala(img_gray, faixa_raio, min_dist_factor=0.8):
    rmin, rmax = faixa_raio
    # param2 baixo para ser mais permissivo com bolhas imperfeitas
    circles = cv2.HoughCircles(
        img_gray, cv2.HOUGH_GRADIENT, dp=1.1, 
        minDist=max(5, int(rmin * min_dist_factor)),
        param1=100, param2=12, 
        minRadius=rmin, maxRadius=rmax
    )
    if circles is None: return []
    return [{"x": float(c[0]), "y": float(c[1]), "r": float(c[2]), "ativo": True} for c in circles[0, :]]

def gerar_candidatos_multicanal(prep: Dict[str, np.ndarray], roi_info: Dict[str, Any], px_per_mm: Optional[float]):
    candidatos = []
    
    # Define faixas de raio automáticas
    h, w = prep["gray_vis"].shape
    base = min(h, w)
    if px_per_mm and px_per_mm > 0:
        faixas = [
            (max(3, int(px_per_mm * 0.005)), max(10, int(px_per_mm * 0.020))), # Pequenas
            (max(10, int(px_per_mm * 0.020)), max(25, int(px_per_mm * 0.050))), # Médias
            (max(25, int(px_per_mm * 0.050)), max(60, int(px_per_mm * 0.150)))  # Grandes
        ]
    else:
        faixas = [(3, 12), (12, 30), (30, 70)]

    # Detecta em múltiplos canais de processamento
    canais = [prep["bilateral_proc"], prep["sharpen_proc"], prep["log_proc"]]
    for canal in canais:
        for faixa in faixas:
            candidatos.extend(detectar_hough_multiescala(canal, faixa))
            
    # Filtra os que estão fora da ROI com folga
    validos = []
    for c in candidatos:
        if ponto_dentro_roi(c["x"], c["y"], roi_info, folga=c["r"] * 0.5):
            validos.append(c)
            
    # Agrupa duplicatas
    return agrupar_candidatos_semelhantes(validos)

def agrupar_candidatos_semelhantes(circles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not circles: return []
    grupos = []
    for c in circles:
        x, y, r = c["x"], c["y"], c["r"]
        encaixou = False
        for g in grupos:
            dist = math.hypot(x - g["x"], y - g["y"])
            r_ref = max(r, g["r"])
            if dist <= 0.7 * r_ref and abs(r - g["r"]) <= 0.5 * r_ref:
                g["xs"].append(x); g["ys"].append(y); g["rs"].append(r)
                g["count"] += 1; encaixou = True; break
        if not encaixou:
            grupos.append({"x": x, "y": y, "r": r, "xs": [x], "ys": [y], "rs": [r], "count": 1, "ativo": True})
    
    finais = []
    for g in grupos:
        finais.append({"x": float(np.mean(g["xs"])), "y": float(np.mean(g["ys"])), "r": float(np.mean(g["rs"])), "ativo": True})
    return finais

# ============================================================
# DESENHO ESTILO REFERÊNCIA (COLORIDO + ID)
# ============================================================
def desenhar_resultado_estilo_referencia(
    img_bgr: np.ndarray, 
    roi_info: Dict[str, Any], 
    circles: List[Dict[str, Any]], 
    titulo: str = "Bolhas Detectadas"
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    
    # 1. Criar sombreamento fora da ROI circular
    overlay = out.copy()
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_roi, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 255, -1)
    overlay[mask_roi == 0] = (20, 20, 20) # Fundo quase preto
    out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    
    # 2. Desenhar contorno da ROI
    cv2.circle(out, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), (255, 255, 255), 2)
    
    # 3. Desenhar bolhas (Coloridas + ID)
    rng = np.random.default_rng(42) # Seed fixa para cores consistentes
    count_ativos = 0
    
    for c in circles:
        if not c.get("ativo", True): continue
        count_ativos += 1
        
        # Cor aleatória vibrante
        color = tuple(int(v) for v in rng.integers(100, 256, size=3))
        cx, cy, r = int(round(c["x"])), int(round(c["y"])), int(round(c["r"]))
        
        # Desenha o contorno da bolha
        cv2.circle(out, (cx, cy), r, color, 2)
        
        # Escreve o ID no centro
        cv2.putText(out, str(count_ativos), (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
    # Título
    cv2.putText(out, f"{titulo}: {count_ativos}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out

# ============================================================
# EDIÇÃO E MEDIÇÃO (MANTIDAS)
# ============================================================
def circles_to_dataframe(circles: List[Dict[str, Any]]) -> pd.DataFrame:
    if not circles: return pd.DataFrame(columns=["ativo", "x", "y", "r"])
    return pd.DataFrame(circles)[["ativo", "x", "y", "r"]]

def dataframe_to_circles(df: pd.DataFrame, roi_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    circles = []
    for _, row in df.iterrows():
        try:
            ativo, x, y, r = bool(row["ativo"]), float(row["x"]), float(row["y"]), float(row["r"])
            if r > 0 and ponto_dentro_roi(x, y, roi_info, folga=r*0.5):
                circles.append({"ativo": ativo, "x": x, "y": y, "r": r})
        except: continue
    return agrupar_candidatos_semelhantes(circles)

def montar_dataframe_medidas(circles: List[Dict[str, Any]], px_per_mm: Optional[float]) -> pd.DataFrame:
    rows = []
    idx = 1
    for c in circles:
        if not c.get("ativo", True): continue
        diam_px = 2.0 * float(c["r"])
        diam_um = (diam_px / px_per_mm) * 1000.0 if px_per_mm and px_per_mm > 0 else None
        rows.append({
            "Bolha": idx, "Centro X (px)": round(c["x"], 1), "Centro Y (px)": round(c["y"], 1),
            "Diâmetro (px)": round(diam_px, 2), "Diâmetro (µm)": round(diam_um, 2) if diam_um else None
        })
        idx += 1
    return pd.DataFrame(rows)

# ============================================================
# TELA PRINCIPAL (RESTRUTURADA)
# ============================================================
def render_consulta_imagens(listar_imagens_supabase, montar_url_publica, session_state):
    garantir_estado(session_state)
    with st.container(border=True):
        st.markdown("## Consulta e Análise Granulométrica")
        if st.button("Atualizar lista de imagens", key="btn_atualizar_lista"):
            imagens, erro = listar_imagens_supabase("")
            if erro: st.error(f"Erro: {erro}")
            else: session_state.lista_imagens_consulta = [img["name"] for img in imagens]
        
        lista = session_state.get("lista_imagens_consulta", [])
        if not lista:
            st.info("Clique em 'Atualizar lista' para carregar as imagens.")
            return
        
        escolhido = st.selectbox("Selecione a imagem", lista, key="select_imagem")
        
        # Carregamento e Calibração
        url_imagem = montar_url_publica(escolhido)
        img_pil = baixar_imagem(url_imagem)
        img_bgr = pil_to_cv(img_pil)
        px_per_mm, img_calibracao, barra_info = detectar_barra_escala_px(img_bgr)
        
        # Definição da ROI Circular Útil
        st.markdown("### 1. Ajuste da Área Útil Circular (ROI)")
        roi_info = obter_roi(session_state, escolhido) or criar_roi_circular_padrao(img_bgr.shape)
        salvar_roi(session_state, escolhido, roi_info)
        
        r1, r2, r3 = st.columns(3)
        roi_cx = r1.number_input("Centro X ROI", min_value=0, max_value=img_bgr.shape[1], value=int(roi_info["cx"]), step=10)
        roi_cy = r2.number_input("Centro Y ROI", min_value=0, max_value=img_bgr.shape[0], value=int(roi_info["cy"]), step=10)
        roi_r = r3.number_input("Raio ROI", min_value=50, max_value=min(img_bgr.shape[:2])//2, value=int(roi_info["r"]), step=10)
        roi_info = {"cx": roi_cx, "cy": roi_cy, "r": roi_r}
        salvar_roi(session_state, escolhido, roi_info)
        
        # Desenha apenas a ROI para ajuste
        img_roi_setup = desenhar_roi_e_circulos(img_bgr, roi_info, [], titulo="Ajuste de ROI")
        st.image(cv_to_pil(img_roi_setup), caption="Área útil circular (ROI)", width=760)
        
        # Exibe Calibração
        st.markdown("### 2. Calibração")
        if px_per_mm: st.success(f"Barra detectada: {px_per_mm:.2f} px = 1,0 mm")
        else: st.warning("Barra de escala não detectada automaticamente.")
        st.image(cv_to_pil(img_calibracao), caption="Detecção da barra", width=380)
        
        # Detecção Automática Otimizada
        st.markdown("### 3. Detecção Automática de Bolhas")
        if st.button("Gerar bolhas automaticamente", key="btn_gerar_aut"):
            mask_roi = criar_mascara_roi(img_bgr.shape, roi_info, barra_info)
            prep = preprocessar_otimizado(img_bgr, mask_roi)
            candidatos = gerar_candidatos_multicanal(prep, roi_info, px_per_mm)
            salvar_circulos(session_state, escolhido, candidatos)
            st.success(f"{len(candidatos)} bolhas detectadas automaticamente.")
            st.rerun()
            
        # Edição Manual (Tabela)
        st.markdown("### 4. Refinamento Manual (Opcional)")
        circles = obter_circulos(session_state, escolhido)
        edited_df = st.data_editor(circles_to_dataframe(circles), use_container_width=True, num_rows="dynamic", key=f"editor_{escolhido}")
        if st.button("Salvar edições", key="btn_salvar_edit"):
            salvar_circulos(session_state, escolhido, dataframe_to_circles(edited_df, roi_info))
            st.success("Edições salvas.")
            st.rerun()
            
        # RESULTADO FINAL ESTILO REFERÊNCIA
        st.markdown("### 5. Resultado Final (Estilo Referência)")
        circles = obter_circulos(session_state, escolhido)
        img_resultado = desenhar_roi_e_circulos(img_bgr, roi_info, circles, titulo="Granulometria de Espuma")
        st.image(cv_to_pil(img_resultado), caption="Bolhas detectadas dentro da área útil circular", width=900)
        
        # Tabela e Métricas (MANTIDAS)
        df_med = montar_dataframe_medidas(circles, px_per_mm)
        if not df_med.empty:
            st.markdown("### 6. Dados Granulométricos")
            st.dataframe(df_med, use_container_width=True)
            
            maiores_500 = df_med[df_med["Diâmetro (µm)"] > 500]
            pct_500 = 100.0 * len(maiores_500) / len(df_med)
            
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total Bolhas", len(df_med))
            s2.metric("Bolhas > 500 µm", len(maiores_500))
            s3.metric("% > 500 µm", f"{pct_500:.2f}%")
            s4.metric("Diâmetro médio (µm)", f"{df_med['Diâmetro (µm)'].mean():.1f}")
            
    except Exception as e:
        st.error(f"Erro no processamento: {e}")
