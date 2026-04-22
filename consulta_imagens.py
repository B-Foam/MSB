import io
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image


# ============================================================
# ESTADO
# ============================================================
def garantir_estado(session_state):
    if "lista_imagens_consulta" not in session_state:
        session_state.lista_imagens_consulta = []

    if "roi_consulta" not in session_state:
        session_state.roi_consulta = {}

    if "imagem_cache" not in session_state:
        session_state.imagem_cache = {}

    if "resultado_cache" not in session_state:
        session_state.resultado_cache = {}


# ============================================================
# DOWNLOAD / CONVERSÃO
# ============================================================
@st.cache_data(show_spinner=False)
def baixar_imagem_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def abrir_imagem(url: str) -> Image.Image:
    data = baixar_imagem_bytes(url)
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    if len(img_cv.shape) == 2:
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


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
    return math.hypot(dx, dy) + r <= roi_info["r"]


# ============================================================
# DETECÇÃO LEVE
# ============================================================
def preprocessar_leve(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray[mask_roi == 0] = 0

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    blackhat = cv2.morphologyEx(
        blur,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blackhat[mask_roi == 0] = 0

    return blackhat


def recortar_roi_para_deteccao(img: np.ndarray, roi_info: Dict[str, int], scale: float = 0.35):
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


def fundir_candidatos(candidatos: List[Dict]) -> List[Dict]:
    if not candidatos:
        return []

    candidatos = sorted(candidatos, key=lambda c: c["r"], reverse=True)
    finais = []

    for c in candidatos:
        manter = True
        for f in finais:
            dist = math.hypot(c["x"] - f["x"], c["y"] - f["y"])
            r_ref = max(c["r"], f["r"])
            if dist < 0.45 * r_ref and abs(c["r"] - f["r"]) < 0.30 * r_ref:
                manter = False
                break
        if manter:
            finais.append(c)

    return finais


def detectar_bolhas_leve(
    img_bgr: np.ndarray,
    roi_info: Dict[str, int],
    mask_roi: np.ndarray,
    px_per_mm: Optional[float],
):
    base = preprocessar_leve(img_bgr, mask_roi)

    crop, (x0, y0), scale = recortar_roi_para_deteccao(base, roi_info, scale=0.35)

    candidatos = []

    if px_per_mm and px_per_mm > 0:
        px_per_mm_small = px_per_mm * scale
        faixas = [
            {
                "minR": max(4, int(px_per_mm_small * 0.020)),
                "maxR": max(11, int(px_per_mm_small * 0.045)),
                "minDist": max(6, int(px_per_mm_small * 0.018)),
                "param2": 14,
            },
            {
                "minR": max(11, int(px_per_mm_small * 0.045)),
                "maxR": max(38, int(px_per_mm_small * 0.140)),
                "minDist": max(10, int(px_per_mm_small * 0.028)),
                "param2": 18,
            },
        ]
    else:
        faixas = [
            {"minR": 4, "maxR": 11, "minDist": 6, "param2": 14},
            {"minR": 11, "maxR": 38, "minDist": 10, "param2": 18},
        ]

    for faixa in faixas:
        circles = cv2.HoughCircles(
            crop,
            cv2.HOUGH_GRADIENT,
            dp=1.25,
            minDist=faixa["minDist"],
            param1=90,
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

            candidatos.append({"x": float(x), "y": float(y), "r": float(r)})

    return fundir_candidatos(candidatos)


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

        # sem baixar/processar mais nada além do essencial
        url = montar_url_publica(imagem_escolhida)
        img_pil = abrir_imagem(url)
        img_bgr = pil_to_cv(img_pil)

        px_per_mm, _, barra_info = detectar_barra_escala_px(img_bgr)

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

        img_roi = desenhar_imagem_roi(img_bgr, roi_info, barra_info=barra_info)

        st.markdown("### Imagem 1 — área útil + calibração")
        st.image(cv_to_pil(img_roi), use_container_width=True)

        if px_per_mm:
            st.success(f"Barra detectada: {px_per_mm:.2f} px para 1,0 mm")
        else:
            st.warning("Barra de calibração não detectada automaticamente.")

        if st.button("Detectar bolhas", key=f"processar_{imagem_escolhida}"):
            with st.spinner("Detectando bolhas..."):
                mask_roi = criar_mascara_roi(img_bgr.shape, roi_info, barra_info)
                bolhas = detectar_bolhas_leve(
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
