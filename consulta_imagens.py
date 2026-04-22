import os
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import streamlit as st


# ============================================================
# UTILITÁRIOS
# ============================================================

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def ler_imagem_url(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem.")
    return img


def _basename(path: str) -> str:
    if not path:
        return "imagem"
    return os.path.basename(path)


def normalizar_lista_imagens(lista_bruta) -> List[Dict]:
    """
    Tenta aceitar diferentes formatos vindos do Supabase/listagem:
    - ["img1.jpg", "img2.jpg"]
    - [{"name": "img1.jpg", "path": "pasta/img1.jpg"}]
    - [{"nome": "...", "caminho": "..."}]
    - [{"path": "..."}]
    """
    itens = []

    if not lista_bruta:
        return itens

    for item in lista_bruta:
        if isinstance(item, str):
            itens.append({
                "nome": _basename(item),
                "path": item,
                "url": None,
            })
            continue

        if isinstance(item, dict):
            nome = (
                item.get("name")
                or item.get("nome")
                or item.get("file_name")
                or item.get("filename")
                or item.get("path")
                or item.get("caminho")
                or item.get("storage_path")
                or item.get("url")
                or "imagem"
            )

            path = (
                item.get("path")
                or item.get("caminho")
                or item.get("storage_path")
                or item.get("name")
                or item.get("nome")
            )

            url = item.get("url")

            itens.append({
                "nome": _basename(nome),
                "path": path,
                "url": url,
            })

    # remove duplicados simples por nome+path
    vistos = set()
    filtrados = []
    for it in itens:
        chave = (it["nome"], it["path"], it["url"])
        if chave not in vistos:
            vistos.add(chave)
            filtrados.append(it)

    return filtrados


def gerar_cores_bgr() -> List[Tuple[int, int, int]]:
    random.seed(123)
    cores = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 200, 80), (255, 80, 200), (80, 255, 255),
        (180, 80, 255), (255, 255, 80), (80, 180, 255),
        (255, 140, 60), (120, 255, 120), (200, 120, 255),
        (255, 120, 170), (120, 255, 220), (220, 220, 220),
        (255, 0, 120), (0, 220, 120), (50, 120, 255),
    ]
    return cores


CORES_BOLHAS = gerar_cores_bgr()


# ============================================================
# ROI CIRCULAR
# ============================================================

def roi_padrao(w: int, h: int) -> Dict[str, int]:
    r = int(min(w, h) * 0.34)
    cx = w // 2
    cy = h // 2
    return {"cx": cx, "cy": cy, "r": r}


def criar_mascara_roi(shape_hw: Tuple[int, int], roi_info: Dict[str, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 255, -1)
    return mask


def ponto_totalmente_dentro_roi(x: float, y: float, r: float, roi_info: Dict[str, int]) -> bool:
    dx = x - roi_info["cx"]
    dy = y - roi_info["cy"]
    dist = math.sqrt(dx * dx + dy * dy)
    return dist + r <= roi_info["r"] - 2


# ============================================================
# BARRA DE ESCALA
# ============================================================

def detectar_barra_escala(img_bgr: np.ndarray) -> Optional[Dict]:
    """
    Detecta a barra verde da imagem no canto inferior esquerdo.
    Retorna:
      {
        "x1","y1","x2","y2","px_per_mm"
      }
    """
    h, w = img_bgr.shape[:2]

    # região típica da barra: parte inferior esquerda
    x0 = 0
    y0 = int(h * 0.72)
    x1 = int(w * 0.38)
    y1 = h

    roi = img_bgr[y0:y1, x0:x1].copy()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # verde brilhante
    lower = np.array([35, 70, 70], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor = None
    melhor_comp = 0

    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 20:
            continue

        # barra horizontal
        if ww < 20:
            continue

        if hh > 20:
            continue

        ratio = ww / max(hh, 1)
        if ratio < 3.5:
            continue

        if ww > melhor_comp:
            melhor_comp = ww
            melhor = (x, y, ww, hh)

    if melhor is None:
        return None

    x, y, ww, hh = melhor

    return {
        "x1": x0 + x,
        "y1": y0 + y + hh // 2,
        "x2": x0 + x + ww,
        "y2": y0 + y + hh // 2,
        "px_per_mm": float(ww),  # assumindo barra = 1,0 mm
    }


# ============================================================
# DESENHO DAS IMAGENS
# ============================================================

def desenhar_roi_e_calibracao(
    img_bgr: np.ndarray,
    roi_info: Dict[str, int],
    barra_info: Optional[Dict],
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # escurece fora da ROI
    mask = criar_mascara_roi((h, w), roi_info)
    escura = (out * 0.45).astype(np.uint8)
    out = np.where(mask[:, :, None] == 255, out, escura)

    # borda ROI
    cv2.circle(
        out,
        (int(roi_info["cx"]), int(roi_info["cy"])),
        int(roi_info["r"]),
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    # barra
    if barra_info is not None:
        x1, y1 = int(barra_info["x1"]), int(barra_info["y1"])
        x2, y2 = int(barra_info["x2"]), int(barra_info["y2"])

        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            "1.0 mm",
            (x1 + 6, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    return out


def desenhar_bolhas_detectadas(
    img_bgr: np.ndarray,
    roi_info: Dict[str, int],
    barra_info: Optional[Dict],
    bolhas: List[Dict],
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    # escurece fora da ROI
    mask = criar_mascara_roi((h, w), roi_info)
    escura = (out * 0.45).astype(np.uint8)
    out = np.where(mask[:, :, None] == 255, out, escura)

    # borda ROI
    cv2.circle(
        out,
        (int(roi_info["cx"]), int(roi_info["cy"])),
        int(roi_info["r"]),
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    # desenha círculos sobre a imagem original
    for i, b in enumerate(bolhas):
        cor = CORES_BOLHAS[i % len(CORES_BOLHAS)]
        x = int(round(b["x"]))
        y = int(round(b["y"]))
        r = int(round(b["r"]))

        esp = 2 if r < 18 else 3
        cv2.circle(out, (x, y), r, cor, esp, lineType=cv2.LINE_AA)

    # barra
    if barra_info is not None:
        x1, y1 = int(barra_info["x1"]), int(barra_info["y1"])
        x2, y2 = int(barra_info["x2"]), int(barra_info["y2"])
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(
            out,
            "1.0 mm",
            (x1 + 6, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    return out


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def preprocessar_primeiro_plano(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray[mask_roi == 0] = 0

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    gray_blur = cv2.bilateralFilter(gray_eq, 9, 40, 40)

    lap = cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)

    sobx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobx, soby)

    focus = 0.55 * lap + 0.45 * grad
    focus = cv2.GaussianBlur(focus, (0, 0), 1.0)
    focus = cv2.normalize(focus, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    focus[mask_roi == 0] = 0

    return {
        "gray": gray,
        "gray_eq": gray_eq,
        "gray_blur": gray_blur,
        "focus": focus,
    }


def recortar_roi_para_deteccao(img_gray: np.ndarray, roi_info: Dict[str, int], scale: float = 0.42):
    h, w = img_gray.shape[:2]

    cx = int(roi_info["cx"])
    cy = int(roi_info["cy"])
    r = int(roi_info["r"])

    margem = 18
    x0 = max(0, cx - r - margem)
    y0 = max(0, cy - r - margem)
    x1 = min(w, cx + r + margem)
    y1 = min(h, cy + r + margem)

    crop = img_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return img_gray.copy(), (0, 0), 1.0

    if scale != 1.0:
        crop_small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        crop_small = crop

    return crop_small, (x0, y0), scale


# ============================================================
# SCORE DE PRIMEIRO PLANO
# ============================================================

def score_primeiro_plano(gray: np.ndarray, focus: np.ndarray, x: float, y: float, r: float):
    x = int(round(x))
    y = int(round(y))
    r = int(round(r))

    if r < 5:
        return None

    h, w = gray.shape[:2]
    pad = int(r * 1.7) + 4

    x0 = max(0, x - pad)
    x1 = min(w, x + pad + 1)
    y0 = max(0, y - pad)
    y1 = min(h, y + pad + 1)

    gray_crop = gray[y0:y1, x0:x1]
    focus_crop = focus[y0:y1, x0:x1]

    if gray_crop.size == 0 or focus_crop.size == 0:
        return None

    yy, xx = np.indices(gray_crop.shape)
    xx = xx + x0
    yy = yy + y0
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    inner = (dist >= 0.22 * r) & (dist <= 0.72 * r)
    ring = (dist >= 0.82 * r) & (dist <= 1.10 * r)
    outer = (dist >= 1.18 * r) & (dist <= 1.52 * r)

    if np.count_nonzero(inner) < 20 or np.count_nonzero(ring) < 20 or np.count_nonzero(outer) < 20:
        return None

    mean_inner = float(np.mean(gray_crop[inner]))
    mean_ring = float(np.mean(gray_crop[ring]))
    mean_outer = float(np.mean(gray_crop[outer]))

    focus_ring = float(np.mean(focus_crop[ring]))
    focus_inner = float(np.mean(focus_crop[inner]))
    focus_outer = float(np.mean(focus_crop[outer]))

    ring_darkness = ((mean_inner + mean_outer) / 2.0) - mean_ring
    focus_gain = focus_ring - (0.5 * focus_inner + 0.5 * focus_outer)
    local_contrast = abs(mean_inner - mean_outer)

    score = (
        1.35 * focus_ring
        + 1.10 * max(0.0, focus_gain)
        + 2.20 * max(0.0, ring_darkness)
        + 0.40 * local_contrast
    )

    return {
        "score": float(score),
        "focus_ring": float(focus_ring),
        "focus_gain": float(focus_gain),
        "ring_darkness": float(ring_darkness),
        "local_contrast": float(local_contrast),
    }


# ============================================================
# PÓS-PROCESSAMENTO DOS CANDIDATOS
# ============================================================

def fundir_candidatos(candidatos: List[Dict]) -> List[Dict]:
    if not candidatos:
        return []

    candidatos = sorted(candidatos, key=lambda c: c["score"], reverse=True)
    finais = []

    for c in candidatos:
        duplicado = False
        for f in finais:
            dx = c["x"] - f["x"]
            dy = c["y"] - f["y"]
            dist = math.sqrt(dx * dx + dy * dy)

            raio_ref = max(c["r"], f["r"])
            if dist < 0.45 * (c["r"] + f["r"]) and abs(c["r"] - f["r"]) < 0.45 * raio_ref:
                duplicado = True
                break

        if not duplicado:
            finais.append(c)

    return finais


def remover_pequenas_dentro_de_grandes(candidatos: List[Dict]) -> List[Dict]:
    if not candidatos:
        return []

    candidatos = sorted(candidatos, key=lambda c: c["r"], reverse=True)
    manter = [True] * len(candidatos)

    for i in range(len(candidatos)):
        if not manter[i]:
            continue
        bi = candidatos[i]

        for j in range(i + 1, len(candidatos)):
            if not manter[j]:
                continue

            bj = candidatos[j]
            dx = bj["x"] - bi["x"]
            dy = bj["y"] - bi["y"]
            dist = math.sqrt(dx * dx + dy * dy)

            # remove bolhas pequenas muito internas
            if bj["r"] < 0.52 * bi["r"] and dist + bj["r"] < 0.92 * bi["r"]:
                manter[j] = False

    return [c for c, ok in zip(candidatos, manter) if ok]


# ============================================================
# DETECÇÃO DAS BOLHAS DE PRIMEIRO PLANO
# ============================================================

def detectar_bolhas_primeiro_plano(
    img_bgr: np.ndarray,
    roi_info: Dict[str, int],
    mask_roi: np.ndarray,
    px_per_mm: Optional[float],
) -> List[Dict]:
    prep = preprocessar_primeiro_plano(img_bgr, mask_roi)

    gray_blur = prep["gray_blur"]
    gray_eq = prep["gray_eq"]
    focus = prep["focus"]

    # detecta em dois mapas: focus e gray_blur
    fontes = [
        ("focus", focus, 0.42),
        ("gray", gray_blur, 0.42),
    ]

    candidatos = []

    for nome_fonte, img_fonte, scale in fontes:
        crop, (x0, y0), scale_used = recortar_roi_para_deteccao(img_fonte, roi_info, scale=scale)

        if px_per_mm and px_per_mm > 0:
            px_small = px_per_mm * scale_used

            faixas = [
                {
                    "nome": "pequenas",
                    "minR": max(5, int(px_small * 0.020)),
                    "maxR": max(11, int(px_small * 0.043)),
                    "minDist": max(8, int(px_small * 0.020)),
                    "param2": 15,
                },
                {
                    "nome": "medias",
                    "minR": max(11, int(px_small * 0.043)),
                    "maxR": max(25, int(px_small * 0.095)),
                    "minDist": max(12, int(px_small * 0.035)),
                    "param2": 16,
                },
                {
                    "nome": "grandes",
                    "minR": max(25, int(px_small * 0.095)),
                    "maxR": max(58, int(px_small * 0.220)),
                    "minDist": max(18, int(px_small * 0.055)),
                    "param2": 17,
                },
            ]
        else:
            faixas = [
                {"nome": "pequenas", "minR": 5, "maxR": 11, "minDist": 8, "param2": 15},
                {"nome": "medias", "minR": 11, "maxR": 25, "minDist": 12, "param2": 16},
                {"nome": "grandes", "minR": 25, "maxR": 58, "minDist": 18, "param2": 17},
            ]

        for faixa in faixas:
            circles = cv2.HoughCircles(
                crop,
                cv2.HOUGH_GRADIENT,
                dp=1.15,
                minDist=faixa["minDist"],
                param1=110,
                param2=faixa["param2"],
                minRadius=faixa["minR"],
                maxRadius=faixa["maxR"],
            )

            if circles is None:
                continue

            circles = np.round(circles[0, :]).astype(int)

            for c in circles:
                xs, ys, rs = int(c[0]), int(c[1]), int(c[2])

                x = xs / scale_used + x0
                y = ys / scale_used + y0
                r = rs / scale_used

                if not ponto_totalmente_dentro_roi(x, y, r, roi_info):
                    continue

                met = score_primeiro_plano(gray_eq, focus, x, y, r)
                if met is None:
                    continue

                # filtros mínimos
                if met["focus_ring"] < 18:
                    continue
                if met["ring_darkness"] < 1.7:
                    continue
                if met["score"] < 22:
                    continue

                candidatos.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "r": float(r),
                        "score": float(met["score"]),
                        "focus_ring": float(met["focus_ring"]),
                        "focus_gain": float(met["focus_gain"]),
                        "ring_darkness": float(met["ring_darkness"]),
                        "local_contrast": float(met["local_contrast"]),
                        "fonte": nome_fonte,
                    }
                )

    candidatos = fundir_candidatos(candidatos)
    candidatos = remover_pequenas_dentro_de_grandes(candidatos)

    if not candidatos:
        return []

    arr_score = np.array([c["score"] for c in candidatos], dtype=np.float32)
    arr_focus = np.array([c["focus_ring"] for c in candidatos], dtype=np.float32)
    arr_dark = np.array([c["ring_darkness"] for c in candidatos], dtype=np.float32)

    # filtro adaptativo: prioriza primeiro plano, mas não mata demais
    thr_score = max(22.0, float(np.percentile(arr_score, 50)))
    thr_focus = max(18.0, float(np.percentile(arr_focus, 45)))
    thr_dark = max(1.7, float(np.percentile(arr_dark, 45)))

    bolhas = [
        c for c in candidatos
        if c["score"] >= thr_score
        and c["focus_ring"] >= thr_focus
        and c["ring_darkness"] >= thr_dark
    ]

    # segunda passada: evita excesso de bolhas pequenas fracas
    bolhas_filtradas = []
    for b in bolhas:
        # regra adicional: bolhas pequenas exigem score melhor
        if b["r"] < 10 and b["score"] < (thr_score + 5):
            continue
        if b["r"] < 8 and b["focus_ring"] < (thr_focus + 4):
            continue
        bolhas_filtradas.append(b)

    bolhas = sorted(
        bolhas_filtradas,
        key=lambda c: (c["score"], c["focus_ring"], c["r"]),
        reverse=True,
    )

    return bolhas


# ============================================================
# RENDER PRINCIPAL
# ============================================================

def render_consulta_imagens(
    listar_imagens_supabase,
    montar_url_publica,
    session_state,
):
    st.subheader("Consulta de imagens")

    try:
        try:
            lista_bruta = listar_imagens_supabase("")
        except TypeError:
            lista_bruta = listar_imagens_supabase()
    except Exception as e:
        st.error(f"Erro ao listar imagens: {e}")
        return

    itens = normalizar_lista_imagens(lista_bruta)

    if not itens:
        st.warning("Nenhuma imagem encontrada.")
        with st.expander("Diagnóstico"):
            st.write("Retorno bruto da listagem:")
            st.write(lista_bruta)
        return

    nomes = [it["nome"] for it in itens]
    nome_sel = st.selectbox("Selecione a imagem", nomes, index=0)

    item_sel = itens[nomes.index(nome_sel)]

    if item_sel.get("url"):
        url_img = item_sel["url"]
    else:
        if not item_sel.get("path"):
            st.error("Não foi possível identificar o caminho da imagem.")
            st.write(item_sel)
            return
        url_img = montar_url_publica(item_sel["path"])

    try:
        img_bgr = ler_imagem_url(url_img)
    except Exception as e:
        st.error(f"Erro ao carregar imagem: {e}")
        st.write("URL usada:", url_img)
        return

    h, w = img_bgr.shape[:2]
    chave_img = item_sel.get("path") or item_sel.get("url") or item_sel["nome"]

    key_roi = f"roi::{chave_img}"
    if key_roi not in session_state:
        session_state[key_roi] = roi_padrao(w, h)

    with st.expander("Ajustar área útil circular", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            session_state[key_roi]["cx"] = st.number_input(
                "Centro X ROI",
                min_value=0,
                max_value=w,
                value=int(session_state[key_roi]["cx"]),
                step=10,
                key=f"cx::{chave_img}",
            )
        with c2:
            session_state[key_roi]["cy"] = st.number_input(
                "Centro Y ROI",
                min_value=0,
                max_value=h,
                value=int(session_state[key_roi]["cy"]),
                step=10,
                key=f"cy::{chave_img}",
            )
        with c3:
            session_state[key_roi]["r"] = st.number_input(
                "Raio ROI",
                min_value=20,
                max_value=max(20, min(w, h)),
                value=int(session_state[key_roi]["r"]),
                step=10,
                key=f"r::{chave_img}",
            )

    roi_info = session_state[key_roi]
    mask_roi = criar_mascara_roi((h, w), roi_info)

    barra_info = detectar_barra_escala(img_bgr)
    px_per_mm = barra_info["px_per_mm"] if barra_info is not None else None

    st.markdown("### Imagem 1 — área útil + calibração")
    img1 = desenhar_roi_e_calibracao(img_bgr, roi_info, barra_info)
    st.image(bgr_to_rgb(img1), use_container_width=True)

    if barra_info is not None:
        st.success(f"Barra detectada: {barra_info['px_per_mm']:.2f} px para 1,0 mm")
    else:
        st.warning("Barra de escala não detectada automaticamente.")

    if st.button("Detectar bolhas", use_container_width=False):
        with st.spinner("Detectando bolhas em primeiro plano..."):
            bolhas = detectar_bolhas_primeiro_plano(
                img_bgr=img_bgr,
                roi_info=roi_info,
                mask_roi=mask_roi,
                px_per_mm=px_per_mm,
            )

        key_bolhas = f"bolhas::{chave_img}"
        session_state[key_bolhas] = bolhas

    key_bolhas = f"bolhas::{chave_img}"
    bolhas = session_state.get(key_bolhas, [])

    if bolhas:
        st.markdown("### Imagem 2 — bolhas detectadas")
        img2 = desenhar_bolhas_detectadas(img_bgr, roi_info, barra_info, bolhas)
        st.image(bgr_to_rgb(img2), use_container_width=True)
        st.info(f"Bolhas detectadas: {len(bolhas)}")
    else:
        st.markdown("### Imagem 2 — bolhas detectadas")
        st.info("Ainda não há bolhas detectadas para esta imagem. Clique em **Detectar bolhas**.")
