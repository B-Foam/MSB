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

    cv2.circle(mask, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 255, -1)

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


def ponto_totalmente_dentro_roi(x: float, y: float, r: float, roi_info: Dict[str, int]) -> bool:
    dx = x - roi_info["cx"]
    dy = y - roi_info["cy"]
    dist = math.hypot(dx, dy)
    return dist + r <= roi_info["r"]


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================
def preprocessar_variantes(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray)

    bilateral = cv2.bilateralFilter(img_clahe, 9, 75, 75)

    blur = cv2.GaussianBlur(bilateral, (0, 0), 1.2)
    sharpen = cv2.addWeighted(bilateral, 1.50, blur, -0.50, 0)

    grad = cv2.morphologyEx(
        sharpen,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    dog = cv2.absdiff(
        cv2.GaussianBlur(sharpen, (0, 0), 1.0),
        cv2.GaussianBlur(sharpen, (0, 0), 2.8),
    )

    blackhat = cv2.morphologyEx(
        sharpen,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    canny = cv2.Canny(sharpen, 35, 110)
    canny = cv2.GaussianBlur(canny, (3, 3), 0)

    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    variantes = {
        "gray": gray,
        "clahe": img_clahe,
        "bilateral": bilateral,
        "sharpen": sharpen,
        "grad": grad,
        "dog": dog,
        "blackhat": blackhat,
        "canny": canny,
    }

    for k in variantes:
        img = variantes[k].copy()
        img[mask_roi == 0] = 0
        variantes[k] = img

    return variantes


# ============================================================
# SCORE DO CÍRCULO
# ============================================================
def score_circulo_borda(ref_img: np.ndarray, x: float, y: float, r: float) -> float:
    x = int(round(x))
    y = int(round(y))
    r = int(round(r))

    if r < 3:
        return 0.0

    h, w = ref_img.shape[:2]

    r_inner = max(1, int(r * 0.50))
    r_ring1 = max(2, int(r * 0.78))
    r_ring2 = max(r_ring1 + 1, int(r * 1.12))
    r_outer1 = max(r_ring2 + 1, int(r * 1.15))
    r_outer2 = max(r_outer1 + 1, int(r * 1.42))

    pad = r_outer2 + 3
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

    centro = dist <= r_inner
    anel = (dist >= r_ring1) & (dist <= r_ring2)
    externo = (dist >= r_outer1) & (dist <= r_outer2)

    if np.count_nonzero(anel) < 8 or np.count_nonzero(centro) < 8 or np.count_nonzero(externo) < 8:
        return 0.0

    mean_centro = float(np.mean(crop[centro]))
    mean_anel = float(np.mean(crop[anel]))
    mean_externo = float(np.mean(crop[externo]))

    score = mean_anel - 0.35 * mean_centro - 0.20 * mean_externo
    return float(score)


# ============================================================
# FUSÃO DE CANDIDATOS
# ============================================================
def fundir_candidatos(candidatos: List[Dict]) -> List[Dict]:
    if not candidatos:
        return []

    candidatos = sorted(candidatos, key=lambda c: c["score"], reverse=True)
    finais = []

    for c in candidatos:
        manter = True
        for f in finais:
            dist = math.hypot(c["x"] - f["x"], c["y"] - f["y"])
            r_ref = max(c["r"], f["r"])

            if dist < 0.35 * r_ref and abs(c["r"] - f["r"]) < 0.28 * r_ref:
                manter = False
                break

        if manter:
            finais.append(c)

    finais = sorted(finais, key=lambda c: c["score"], reverse=True)

    if len(finais) > 1800:
        finais = finais[:1800]

    return finais


# ============================================================
# DETECÇÃO AGRESSIVA
# ============================================================
def detectar_bolhas_multiescala(
    img_bgr: np.ndarray,
    roi_info: Dict[str, int],
    mask_roi: np.ndarray,
    px_per_mm: Optional[float],
):
    variantes = preprocessar_variantes(img_bgr, mask_roi)

    ref_score = cv2.addWeighted(variantes["grad"], 0.40, variantes["blackhat"], 0.60, 0)

    candidatos = []

    if px_per_mm and px_per_mm > 0:
        faixas = [
            {"nome": "micro",   "minR": max(3, int(px_per_mm * 0.005)), "maxR": max(8,  int(px_per_mm * 0.020)), "minDist": max(4,  int(px_per_mm * 0.008)), "param2": 6},
            {"nome": "pequena", "minR": max(6, int(px_per_mm * 0.018)), "maxR": max(13, int(px_per_mm * 0.035)), "minDist": max(6,  int(px_per_mm * 0.014)), "param2": 7},
            {"nome": "media",   "minR": max(10, int(px_per_mm * 0.030)), "maxR": max(28, int(px_per_mm * 0.080)), "minDist": max(8, int(px_per_mm * 0.022)), "param2": 8},
            {"nome": "grande",  "minR": max(22, int(px_per_mm * 0.070)), "maxR": max(80, int(px_per_mm * 0.220)), "minDist": max(14, int(px_per_mm * 0.045)), "param2": 10},
        ]
    else:
        faixas = [
            {"nome": "micro",   "minR": 3,  "maxR": 8,  "minDist": 4,  "param2": 6},
            {"nome": "pequena", "minR": 6,  "maxR": 13, "minDist": 6,  "param2": 7},
            {"nome": "media",   "minR": 10, "maxR": 28, "minDist": 8,  "param2": 8},
            {"nome": "grande",  "minR": 22, "maxR": 80, "minDist": 14, "param2": 10},
        ]

    bases = [
        variantes["gray"],
        variantes["clahe"],
        variantes["bilateral"],
        variantes["sharpen"],
        variantes["dog"],
        variantes["blackhat"],
        variantes["canny"],
    ]

    for base in bases:
        for faixa in faixas:
            circles = cv2.HoughCircles(
                base,
                cv2.HOUGH_GRADIENT,
                dp=1.08,
                minDist=faixa["minDist"],
                param1=85,
                param2=faixa["param2"],
                minRadius=faixa["minR"],
                maxRadius=faixa["maxR"],
            )

            if circles is None:
                continue

            circles = np.round(circles[0, :]).astype(int)

            for c in circles:
                x, y, r = int(c[0]), int(c[1]), int(c[2])

                if not ponto_totalmente_dentro_roi(x, y, r, roi_info):
                    continue

                score = score_circulo_borda(ref_score, x, y, r)

                if score < 1.2:
                    continue

                candidatos.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "r": float(r),
                        "score": float(score),
                    }
                )

    bolhas = fundir_candidatos(candidatos)

    if len(bolhas) > 0:
        scores = np.array([b["score"] for b in bolhas], dtype=float)
        limiar = max(1.5, float(np.percentile(scores, 8)))
        bolhas = [b for b in bolhas if b["score"] >= limiar]

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
