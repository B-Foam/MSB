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
    img_annot = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    x0 = 0
    x1 = int(w * 0.35)
    y0 = int(h * 0.78)
    y1 = h

    crop = img_bgr[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

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
# ROI
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

    cv2.circle(mask, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 255, -1)

    topo = int(h * 0.045)
    mask[:topo, :] = 0

    if barra_info is not None:
        x, y, ww, hh = barra_info["x"], barra_info["y"], barra_info["w"], barra_info["h"]
        x_ini = max(0, x - 20)
        y_ini = max(0, y - 50)
        x_fim = min(w, x + ww + 90)
        y_fim = min(h, y + hh + 20)
        mask[y_ini:y_fim, x_ini:x_fim] = 0

    return mask


def ponto_totalmente_dentro_roi(x: float, y: float, r: float, roi_info: Dict[str, int]) -> bool:
    dx = x - roi_info["cx"]
    dy = y - roi_info["cy"]
    dist = math.hypot(dx, dy)
    return dist + r <= roi_info["r"]


# ============================================================
# PRÉ-PROCESSAMENTO RÁPIDO
# ============================================================
def preprocessar_rapido(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray[mask_roi == 0] = 0

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    bilateral = cv2.bilateralFilter(clahe_img, 7, 60, 60)

    blackhat = cv2.morphologyEx(
        bilateral,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(blackhat, 28, 255, cv2.THRESH_BINARY)
    mask[mask_roi == 0] = 0
    blackhat[mask_roi == 0] = 0

    return {
        "blackhat": blackhat,
        "mask": mask,
    }


# ============================================================
# CROP DA ROI PARA DETECÇÃO
# ============================================================
def recortar_roi_para_deteccao(img: np.ndarray, roi_info: Dict[str, int], scale: float = 0.5):
    cx, cy, r = roi_info["cx"], roi_info["cy"], roi_info["r"]

    x0 = max(0, int(cx - r))
    y0 = max(0, int(cy - r))
    x1 = min(img.shape[1], int(cx + r))
    y1 = min(img.shape[0], int(cy + r))

    crop = img[y0:y1, x0:x1].copy()

    if scale != 1.0:
        crop_small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        crop_small = crop

    return crop_small, (x0, y0), scale


# ============================================================
# SCORE SIMPLES
# ============================================================
def score_circulo_rapido(ref_img: np.ndarray, x: float, y: float, r: float) -> float:
    x = int(round(x))
    y = int(round(y))
    r = int(round(r))

    if r < 3:
        return 0.0

    h, w = ref_img.shape[:2]
    pad = int(r * 1.4) + 3

    x0 = max(0, x - pad)
    x1 = min(w, x + pad + 1)
    y0 = max(0, y - pad)
    y1 = min(h, y + pad + 1)

    crop = ref_img[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0

    yy, xx = np.indices(crop.shape)
    xx = xx + x0
    yy = yy + y0
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    anel = (dist >= int(r * 0.80)) & (dist <= int(r * 1.15))
    centro = dist <= int(r * 0.55)

    if np.count_nonzero(anel) < 8 or np.count_nonzero(centro) < 8:
        return 0.0

    mean_anel = float(np.mean(crop[anel]))
    mean_centro = float(np.mean(crop[centro]))
    return mean_anel - 0.28 * mean_centro


# ============================================================
# FUSÃO MAIS RÁPIDA
# ============================================================
def fundir_candidatos_rapido(candidatos: List[Dict]) -> List[Dict]:
    if not candidatos:
        return []

    candidatos = sorted(candidatos, key=lambda c: c["r"], reverse=True)
    finais = []

    for c in candidatos:
        manter = True
        for f in finais:
            dist = math.hypot(c["x"] - f["x"], c["y"] - f["y"])
            r_ref = max(c["r"], f["r"])
            if dist < 0.30 * r_ref and abs(c["r"] - f["r"]) < 0.25 * r_ref:
                manter = False
                break
        if manter:
            finais.append(c)

    return finais


# ============================================================
# DETECÇÃO MAIS RÁPIDA
# ============================================================
def detectar_bolhas_multiescala(
    img_bgr: np.ndarray,
    roi_info: Dict[str, int],
    mask_roi: np.ndarray,
    px_per_mm: Optional[float],
):
    proc = preprocessar_rapido(img_bgr, mask_roi)

    base = proc["blackhat"]

    # scale menor = mais rápido
    crop, (x0, y0), scale = recortar_roi_para_deteccao(base, roi_info, scale=0.4)

    candidatos = []

    if px_per_mm and px_per_mm > 0:
        px_per_mm_small = px_per_mm * scale
        faixas = [
            {
                "minR": max(3, int(px_per_mm_small * 0.010)),
                "maxR": max(10, int(px_per_mm_small * 0.035)),
                "minDist": max(5, int(px_per_mm_small * 0.012)),
                "param2": 7,
            },
            {
                "minR": max(10, int(px_per_mm_small * 0.030)),
                "maxR": max(45, int(px_per_mm_small * 0.140)),
                "minDist": max(8, int(px_per_mm_small * 0.022)),
                "param2": 9,
            },
        ]
    else:
        faixas = [
            {"minR": 3, "maxR": 10, "minDist": 5, "param2": 7},
            {"minR": 10, "maxR": 45, "minDist": 8, "param2": 9},
        ]

    for faixa in faixas:
        circles = cv2.HoughCircles(
            crop,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=faixa["minDist"],
            param1=80,
            param2=faixa["param2"],
            minRadius=faixa["minR"],
            maxRadius=faixa["maxR"],
        )

        if circles is None:
            continue

        circles = np.round(circles[0, :]).astype(int)

        for c in circles:
            xs, ys, rs = int(c[0]), int(c[1]), int(c[2])

            x = xs / scale + x0
            y = ys / scale + y0
            r = rs / scale

            if not ponto_totalmente_dentro_roi(x, y, r, roi_info):
                continue

            score = score_circulo_rapido(base, x, y, r)
            if score < 0.8:
                continue

            candidatos.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "r": float(r),
                    "score": 1.0,
                }
            )

    bolhas = fundir_candidatos_rapido(candidatos)
    return bolhas


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

    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (245, 245, 245), -1)

    rng = np.random.default_rng(123)

    for i, b in enumerate(bolhas, start=1):
        x = int(round(b["x"]))
        y = int(round(b["y"]))
        r = int(round(b["r"]))

        color = tuple(int(v) for v in rng.integers(30, 256, size=3))
        cv2.circle(out, (x, y), r, color, -1)

        thickness = max(1, int(round(r * 0.10)))
        cv2.circle(out, (x, y), r, (0, 0, 0), thickness)

        if r >= 11:
            cv2.putText(
                out,
                str(i),
                (x - int(r * 0.35), y + int(r * 0.15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), 2)

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
                bolhas = detectar_bolhas_multiescala(
                    img_bgr=img_bgr,
                    roi_info=roi_info,
                    mask_roi=mask_roi,
                    px_per_mm=px_per_mm,
                )

                img_final = desenhar_bolhas_coloridas(
                    shape=img_bgr.shape,
                    roi_info=roi_info,
                    bolhas=bolhas,
                    barra_info=barra_info,
                )

                st.markdown("### Imagem 2 — bolhas detectadas")
                st.image(cv_to_pil(img_final), use_container_width=True)
                st.info(f"Bolhas detectadas: {len(bolhas)}")
