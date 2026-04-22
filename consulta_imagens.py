import io
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image


# ============================================================
# UTILITÁRIOS
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


def garantir_estado(session_state):
    if "lista_imagens_consulta" not in session_state:
        session_state.lista_imagens_consulta = []
    if "roi_consulta" not in session_state:
        session_state.roi_consulta = {}


# ============================================================
# BARRA DE ESCALA
# ============================================================
def detectar_barra_escala_px(img_bgr: np.ndarray):
    """
    Detecta a barra de escala na região inferior esquerda.
    Retorna:
      px_per_mm, imagem_anotada, barra_info
    """
    img_annot = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    # Região de busca inferior esquerda
    x0 = 0
    x1 = int(w * 0.35)
    y0 = int(h * 0.78)
    y1 = h

    crop = img_bgr[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # procurar elementos escuros
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

        # barra horizontal longa
        if ww > 60 and hh < 25 and aspect > 4.5 and area > melhor_area:
            melhor_area = area
            melhor = (x, y, ww, hh)

    if melhor is None:
        return None, img_annot, None

    x, y, ww, hh = melhor
    gx = x0 + x
    gy = y0 + y

    cv2.rectangle(img_annot, (gx, gy), (gx + ww, gy + hh), (0, 255, 0), 2)
    cv2.line(img_annot, (gx, gy + hh // 2), (gx + ww, gy + hh // 2), (0, 255, 0), 2)
    cv2.putText(
        img_annot,
        f"1.0 mm = {ww} px",
        (gx, max(20, gy - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    barra_info = {"x": gx, "y": gy, "w": ww, "h": hh}
    return float(ww), img_annot, barra_info


# ============================================================
# ROI CIRCULAR
# ============================================================
def criar_roi_padrao(shape: Tuple[int, int, int]):
    h, w = shape[:2]
    cx = w // 2
    cy = h // 2
    r = int(min(h, w) * 0.38)
    return {"cx": cx, "cy": cy, "r": r}


def criar_mascara_roi(shape, roi_info: Dict[str, int], barra_info=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.circle(
        mask,
        (int(roi_info["cx"]), int(roi_info["cy"])),
        int(roi_info["r"]),
        255,
        -1,
    )

    # remove cabeçalho superior
    topo = int(h * 0.045)
    mask[:topo, :] = 0

    # remove região da barra
    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        x_ini = max(0, x - 20)
        y_ini = max(0, y - 50)
        x_fim = min(w, x + ww + 90)
        y_fim = min(h, y + hh + 20)
        mask[y_ini:y_fim, x_ini:x_fim] = 0

    return mask


# ============================================================
# SEGMENTAÇÃO POR WATERSHED (CORREÇÃO DO RUÍDO)
# ============================================================
def preprocessar_para_watershed(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Filtro Bilateral mais agressivo para suavizar a textura interna das bolhas
    bilateral = cv2.bilateralFilter(gray, 15, 80, 80)

    # 2. Threshold Adaptativo (Invertido, pois paredes são escuras)
    # A área útil é isolada pela máscara ROI
    img_roi = cv2.bitwise_and(bilateral, bilateral, mask=mask_roi)
    thresh = cv2.adaptiveThreshold(img_roi, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5)

    # 3. Limpeza morfológica para remover ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    return gray, opening


def detectar_bolhas_watershed(img_gray, opening, mask_roi, roi_info):
    # 4. Transformada de distância (separa bolhas conectadas)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # 5. Threshold nos marcadores (Aqui definimos o centro de cada bolha)
    # O valor 0.35 é o "foco". Aumente para separar mais, diminua para pegar menores.
    _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)

    # 6. Watershed
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    
    # Adicionamos uma "fronteira" para evitar que o Watershed "fuja"
    cv2.circle(markers, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 0, 2)
    
    # Região desconhecida (fundo da ROI) = 0
    markers[opening == 0] = 0

    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(img_bgr, markers)

    # Extrair centros e raios a partir dos marcadores
    candidatos = []
    for label in np.unique(labels):
        if label <= 1: continue

        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        mask[labels == label] = 255

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) > 20: # Filtro de área mínima
                (x, y), r = cv2.minEnclosingCircle(cnt)
                candidatos.append({"x": x, "y": y, "r": r, "score": 1.0})

    return candidatos


# ============================================================
# DESENHOS
# ============================================================
def desenhar_imagem_roi(
    img_bgr: np.ndarray,
    roi_info: Dict[str, int],
    barra_info=None,
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    overlay = out.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (roi_info["cx"], roi_info["cy"]), roi_info["r"], 255, -1)
    overlay[mask == 0] = (25, 25, 25)
    out = cv2.addWeighted(out, 0.62, overlay, 0.38, 0)

    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), 2)

    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        cv2.line(out, (x, y + hh // 2), (x + ww, y + hh // 2), (0, 255, 0), 2)

    return out


def desenhar_bolhas_coloridas(
    shape,
    roi_info: Dict[str, int],
    bolhas: List[Dict],
    barra_info=None,
) -> np.ndarray:
    h, w = shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # fundo preto + área útil branca
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), -1)

    rng = np.random.default_rng(123)

    for i, b in enumerate(bolhas, start=1):
        x = int(round(b["x"]))
        y = int(round(b["y"]))
        r = int(round(b["r"]))

        color = tuple(int(v) for v in rng.integers(30, 256, size=3))

        # preenchimento
        cv2.circle(out, (x, y), r, color, -1)

        # contorno preto
        thickness = max(1, int(round(r * 0.10)))
        cv2.circle(out, (x, y), r, (0, 0, 0), thickness)

        # número apenas para bolhas maiores
        if r >= 11:
            txt = str(i)
            cv2.putText(
                out,
                txt,
                (x - int(r * 0.35), y + int(r * 0.15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # contorno externo da ROI
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), 2)

    # desenhar barra de escala
    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
        cv2.line(out, (x, y + hh // 2), (x + ww, y + hh // 2), (0, 255, 0), 2)
        cv2.putText(
            out,
            "1.0 mm",
            (x + ww + 10, y + hh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out


# ============================================================
# RENDER PRINCIPAL
# ============================================================
def render_consulta_imagens(listar_imagens_supabase, montar_url_publica, session_state):
    garantir_estado(session_state)

    with st.container(border=True):
        st.markdown("## Consulta de imagens")

        if st.button("Atualizar lista", key="btn_atualizar_lista_consulta"):
            imagens, erro = listar_imagens_supabase("")
            if erro:
                st.error(f"Erro ao listar imagens: {erro}")
                return
            session_state.lista_imagens_consulta = [img["name"] for img in imagens]

        lista = session_state.get("lista_imagens_consulta", [])
        if not lista:
            st.info("Clique em 'Atualizar lista' para carregar as imagens.")
            return

        imagem_escolhida = st.selectbox(
            "Selecione a imagem",
            lista,
            key="select_imagem_consulta",
        )

        url = montar_url_publica(imagem_escolhida)
        img_pil = baixar_imagem(url)
        img_bgr = pil_to_cv(img_pil)

        px_per_mm, img_barra, barra_info = detectar_barra_escala_px(img_bgr)

        # ROI por imagem
        if imagem_escolhida not in session_state.roi_consulta:
            session_state.roi_consulta[imagem_escolhida] = criar_roi_padrao(img_bgr.shape)

        roi_info = session_state.roi_consulta[imagem_escolhida]

        st.markdown("### Ajuste da área útil circular")
        col1, col2, col3 = st.columns(3)
        with col1:
            roi_info["cx"] = st.number_input(
                "Centro X ROI",
                min_value=0,
                max_value=int(img_bgr.shape[1]),
                value=int(roi_info["cx"]),
                step=1,
                key=f"roi_cx_{imagem_escolhida}",
            )
        with col2:
            roi_info["cy"] = st.number_input(
                "Centro Y ROI",
                min_value=0,
                max_value=int(img_bgr.shape[0]),
                value=int(roi_info["cy"]),
                step=1,
                key=f"roi_cy_{imagem_escolhida}",
            )
        with col3:
            roi_info["r"] = st.number_input(
                "Raio ROI",
                min_value=10,
                max_value=int(min(img_bgr.shape[:2])),
                value=int(roi_info["r"]),
                step=1,
                key=f"roi_r_{imagem_escolhida}",
            )

        session_state.roi_consulta[imagem_escolhida] = roi_info

        mask_roi = criar_mascara_roi(img_bgr.shape, roi_info, barra_info)

        st.markdown("### Imagem 1 — área útil + calibração")
        img_roi = desenhar_imagem_roi(img_bgr, roi_info, barra_info=barra_info)
        st.image(cv_to_pil(img_roi), use_container_width=True)

        if px_per_mm:
            st.success(f"Barra detectada: {px_per_mm:.2f} px para 1,0 mm")
        else:
            st.warning("Barra de calibração não detectada automaticamente.")

        processar = st.button("Detectar bolhas", key=f"processar_{imagem_escolhida}")

        if processar:
            with st.spinner("Detectando bolhas..."):
                img_gray, opening = preprocessar_para_watershed(img_bgr, mask_roi)
                bolhas = detectar_bolhas_watershed(img_gray, opening, mask_roi, roi_info)

                img_final = desenhar_bolhas_coloridas(
                    shape=img_bgr.shape,
                    roi_info=roi_info,
                    bolhas=bolhas,
                    barra_info=barra_info,
                )

                st.markdown("### Imagem 2 — bolhas detectadas")
                st.image(cv_to_pil(img_final), use_container_width=True)
                st.info(f"Bolhas detectadas: {len(bolhas)}")
