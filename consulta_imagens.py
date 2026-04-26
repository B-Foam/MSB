import io
import math
import os
import json
from typing import Dict, Optional, Tuple, List, Callable, Any

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


# ============================================================
# CACHE / UTILITÁRIOS
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


def garantir_estado(session_state):
    if "lista_imagens_consulta" not in session_state:
        session_state.lista_imagens_consulta = []
    if "roi_consulta" not in session_state:
        session_state.roi_consulta = {}
    if "resultados_testes_granulometria" not in session_state:
        session_state.resultados_testes_granulometria = []


def extrair_tag_teste(nome_arquivo: str) -> str:
    return os.path.splitext(os.path.basename(nome_arquivo))[0]


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
# PRÉ-PROCESSAMENTO LEVE
# ============================================================
def preprocessar_leve(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray[mask_roi == 0] = 0

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
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


# ============================================================
# CROP DA ROI
# ============================================================
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


# ============================================================
# SCORE MAIS SELETIVO
# ============================================================
def score_circulo(ref_img: np.ndarray, x: float, y: float, r: float) -> float:
    x = int(round(x))
    y = int(round(y))
    r = int(round(r))

    if r < 4:
        return 0.0

    h, w = ref_img.shape[:2]
    pad = int(r * 1.5) + 3

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

    centro = dist <= int(r * 0.55)
    anel = (dist >= int(r * 0.82)) & (dist <= int(r * 1.12))
    externo = (dist >= int(r * 1.18)) & (dist <= int(r * 1.45))

    if np.count_nonzero(anel) < 10 or np.count_nonzero(centro) < 10 or np.count_nonzero(externo) < 10:
        return 0.0

    mean_anel = float(np.mean(crop[anel]))
    mean_centro = float(np.mean(crop[centro]))
    mean_externo = float(np.mean(crop[externo]))

    return mean_anel - 0.45 * mean_centro - 0.25 * mean_externo


# ============================================================
# FILTROS GEOMÉTRICOS
# ============================================================
def remover_pequenas_dentro_de_grandes(candidatos: List[Dict]) -> List[Dict]:
    if not candidatos:
        return []

    candidatos = sorted(candidatos, key=lambda c: c["r"], reverse=True)
    finais = []

    for c in candidatos:
        manter = True
        for f in finais:
            dist = math.hypot(c["x"] - f["x"], c["y"] - f["y"])

            if c["r"] < 0.55 * f["r"] and dist + c["r"] < 0.82 * f["r"]:
                manter = False
                break

        if manter:
            finais.append(c)

    return finais


def fundir_candidatos(candidatos: List[Dict]) -> List[Dict]:
    if not candidatos:
        return []

    candidatos = sorted(candidatos, key=lambda c: (c["score"], c["r"]), reverse=True)
    finais = []

    for c in candidatos:
        manter = True
        for f in finais:
            dist = math.hypot(c["x"] - f["x"], c["y"] - f["y"])
            r_ref = max(c["r"], f["r"])

            if dist < 0.55 * r_ref and abs(c["r"] - f["r"]) < 0.35 * r_ref:
                manter = False
                break

        if manter:
            finais.append(c)

    return finais


# ============================================================
# DETECÇÃO AJUSTADA
# ============================================================
def detectar_bolhas_leve(
    img_bgr: np.ndarray,
    roi_info: Dict[str, int],
    mask_roi: np.ndarray,
    px_per_mm: Optional[float],
    param2_pequenas: int = 15,
    score_min_pequenas: float = 5.5,
    param2_medias_grandes: int = 14,
    score_min_medias_grandes: float = 4.0,
):
    base = preprocessar_leve(img_bgr, mask_roi)

    crop, (x0, y0), scale = recortar_roi_para_deteccao(base, roi_info, scale=0.35)

    candidatos = []

    if px_per_mm and px_per_mm > 0:
        px_per_mm_small = px_per_mm * scale

        faixas = [
            {
                "nome": "pequenas",
                "minR": max(5, int(px_per_mm_small * 0.025)),
                "maxR": max(10, int(px_per_mm_small * 0.045)),
                "minDist": max(8, int(px_per_mm_small * 0.022)),
                "param2": param2_pequenas,
            },
            {
                "nome": "medias_grandes",
                "minR": max(10, int(px_per_mm_small * 0.045)),
                "maxR": max(46, int(px_per_mm_small * 0.180)),
                "minDist": max(12, int(px_per_mm_small * 0.040)),
                "param2": param2_medias_grandes,
            },
        ]
    else:
        faixas = [
            {"nome": "pequenas", "minR": 5, "maxR": 10, "minDist": 8, "param2": param2_pequenas},
            {"nome": "medias_grandes", "minR": 10, "maxR": 46, "minDist": 12, "param2": param2_medias_grandes},
        ]

    for faixa in faixas:
        circles = cv2.HoughCircles(
            crop,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
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

            score = score_circulo(base, x, y, r)

            if faixa["nome"] == "pequenas":
                if score < score_min_pequenas:
                    continue
            else:
                if score < score_min_medias_grandes:
                    continue

            candidatos.append(
                {
                    "id": len(candidatos) + 1,
                    "x": float(x),
                    "y": float(y),
                    "r": float(r),
                    "score": float(score),
                    "grupo": faixa["nome"],
                }
            )

    bolhas = fundir_candidatos(candidatos)
    bolhas = remover_pequenas_dentro_de_grandes(bolhas)

    if len(bolhas) > 0:
        scores = np.array([b["score"] for b in bolhas], dtype=float)
        limiar = max(5.0, float(np.percentile(scores, 25)))
        bolhas = [b for b in bolhas if b["score"] >= limiar]

    for i, b in enumerate(bolhas, start=1):
        b["id"] = i

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
# TRATAMENTO DOS RESULTADOS
# ============================================================
def montar_tabela_bolhas(bolhas: List[Dict], px_per_mm: Optional[float]) -> pd.DataFrame:
    registros = []

    for b in bolhas:
        diametro_px = 2.0 * float(b["r"])

        if px_per_mm and px_per_mm > 0:
            diametro_mm = diametro_px / float(px_per_mm)
            diametro_um = diametro_mm * 1000.0
        else:
            diametro_mm = np.nan
            diametro_um = np.nan

        registros.append({
            "id": b.get("id"),
            "x_px": round(float(b["x"]), 2),
            "y_px": round(float(b["y"]), 2),
            "raio_px": round(float(b["r"]), 2),
            "diametro_px": round(diametro_px, 2),
            "diametro_mm": round(diametro_mm, 6) if pd.notna(diametro_mm) else np.nan,
            "diametro_um": round(diametro_um, 2) if pd.notna(diametro_um) else np.nan,
            "score": round(float(b.get("score", np.nan)), 4),
            "grupo": b.get("grupo", ""),
            "maior_500_um": bool(diametro_um > 500.0) if pd.notna(diametro_um) else False,
        })

    df = pd.DataFrame(registros)

    if not df.empty:
        df = df.sort_values(by="diametro_um", ascending=False, na_position="last").reset_index(drop=True)

    return df


def montar_tabela_faixas(df_bolhas: pd.DataFrame) -> pd.DataFrame:
    if df_bolhas.empty or "diametro_um" not in df_bolhas.columns:
        return pd.DataFrame(columns=["faixa_um", "quantidade"])

    bins = [0, 100, 200, 300, 400, 500, 1000, 2000, np.inf]
    labels = [
        "0–100",
        "100–200",
        "200–300",
        "300–400",
        "400–500",
        "500–1000",
        "1000–2000",
        ">2000",
    ]

    serie = df_bolhas["diametro_um"].dropna()
    categorias = pd.cut(serie, bins=bins, labels=labels, right=False, include_lowest=True)
    contagem = categorias.value_counts().reindex(labels, fill_value=0)

    df_faixas = pd.DataFrame({
        "faixa_um": labels,
        "quantidade": contagem.values,
    })

    return df_faixas


def dataframe_para_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


def montar_payload_resultado(
    tag_teste: str,
    percentual_maior_500: float,
    quantidade_total: int,
    df_bolhas: pd.DataFrame,
    df_faixas: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "tag_teste": tag_teste,
        "percentual_bolhas_maiores_500_um": round(float(percentual_maior_500), 4),
        "quantidade_total_bolhas": int(quantidade_total),
        "grafico_barras_resultados": df_faixas.to_dict(orient="records"),
        "tabela_bolhas": df_bolhas.to_dict(orient="records"),
    }


# ============================================================
# RENDER PRINCIPAL
# ============================================================
def render_consulta_imagens(
    listar_imagens_supabase,
    montar_url_publica,
    session_state,
    salvar_resultado_teste: Optional[Callable[[Dict[str, Any]], Any]] = None,
):
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

        st.markdown("### Ajustes de sensibilidade")
        c1, c2 = st.columns(2)

        with c1:
            param2_pequenas = st.slider(
                "param2 — bolhas pequenas",
                min_value=8,
                max_value=25,
                value=15,
                step=1,
                help="Maior valor = menos bolhas pequenas detectadas",
                key=f"param2_peq_{imagem_escolhida}",
            )

            score_min_pequenas = st.slider(
                "score mínimo — bolhas pequenas",
                min_value=2.0,
                max_value=10.0,
                value=5.5,
                step=0.1,
                help="Maior valor = filtro mais rígido para bolhas pequenas",
                key=f"score_peq_{imagem_escolhida}",
            )

        with c2:
            param2_medias_grandes = st.slider(
                "param2 — bolhas médias/grandes",
                min_value=8,
                max_value=25,
                value=14,
                step=1,
                help="Menor valor = mais bolhas médias/grandes detectadas",
                key=f"param2_medg_{imagem_escolhida}",
            )

            score_min_medias_grandes = st.slider(
                "score mínimo — bolhas médias/grandes",
                min_value=2.0,
                max_value=10.0,
                value=4.0,
                step=0.1,
                help="Menor valor = mais bolhas médias/grandes aceitas",
                key=f"score_medg_{imagem_escolhida}",
            )

        mask_roi = criar_mascara_roi(img_bgr.shape, roi_info, barra_info)

        if px_per_mm:
            st.success(f"Barra detectada: {px_per_mm:.2f} px para 1,0 mm")
        else:
            st.warning("Barra de calibração não detectada automaticamente.")

        key_bolhas = f"bolhas_detectadas::{imagem_escolhida}"

        if st.button("Detectar bolhas", key=f"processar_{imagem_escolhida}"):
            with st.spinner("Detectando bolhas..."):
                bolhas = detectar_bolhas_leve(
                    img_bgr=img_bgr,
                    roi_info=roi_info,
                    mask_roi=mask_roi,
                    px_per_mm=px_per_mm,
                    param2_pequenas=param2_pequenas,
                    score_min_pequenas=score_min_pequenas,
                    param2_medias_grandes=param2_medias_grandes,
                    score_min_medias_grandes=score_min_medias_grandes,
                )
                session_state[key_bolhas] = bolhas

        bolhas = session_state.get(key_bolhas, [])

        # ============================================================
        # VISUALIZAÇÃO COMPARATIVA LADO A LADO
        # ============================================================
        st.markdown("### Comparação visual da análise")

        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.markdown("**Imagem original com ROI e calibração**")
            img_roi = desenhar_imagem_roi(
                img_bgr,
                roi_info,
                barra_info=barra_info,
            )
            st.image(
                cv_to_pil(img_roi),
                use_container_width=True,
            )

        with col_img2:
            st.markdown("**Imagem tratada com bolhas detectadas**")

            if bolhas:
                img_final = desenhar_bolhas_coloridas(
                    shape=img_bgr.shape,
                    roi_info=roi_info,
                    bolhas=bolhas,
                    barra_info=barra_info,
                )
                st.image(
                    cv_to_pil(img_final),
                    use_container_width=True,
                )
            else:
                st.info("Ainda não há bolhas detectadas. Clique em **Detectar bolhas**.")

        # ============================================================
        # RESULTADOS NUMÉRICOS
        # ============================================================
        if bolhas:
            df_bolhas = montar_tabela_bolhas(bolhas, px_per_mm)
            df_faixas = montar_tabela_faixas(df_bolhas)

            quantidade_total = int(len(df_bolhas))
            quantidade_maior_500 = int(df_bolhas["maior_500_um"].sum()) if not df_bolhas.empty else 0
            percentual_maior_500 = (100.0 * quantidade_maior_500 / quantidade_total) if quantidade_total > 0 else 0.0

            st.markdown("### Resumo dos resultados")
            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Quantidade total", quantidade_total)

            with m2:
                st.metric("Bolhas > 500 µm", quantidade_maior_500)

            with m3:
                st.metric("% > 500 µm", f"{percentual_maior_500:.2f}%")

            st.markdown("### Gráfico de barras — distribuição por faixa (µm)")
            st.bar_chart(df_faixas.set_index("faixa_um"))

            st.markdown("### Tabela do gráfico de barras")
            st.dataframe(df_faixas, use_container_width=True)

            st.markdown("### Tabela de bolhas")
            st.dataframe(df_bolhas, use_container_width=True)

            csv_bolhas = dataframe_para_csv_bytes(df_bolhas)
            tag_teste = extrair_tag_teste(imagem_escolhida)

            st.download_button(
                label="Baixar tabela de bolhas (CSV)",
                data=csv_bolhas,
                file_name=f"{tag_teste}_tabela_bolhas.csv",
                mime="text/csv",
                key=f"download_csv_bolhas_{imagem_escolhida}",
            )

            payload_resultado = montar_payload_resultado(
                tag_teste=tag_teste,
                percentual_maior_500=percentual_maior_500,
                quantidade_total=quantidade_total,
                df_bolhas=df_bolhas,
                df_faixas=df_faixas,
            )

            if st.button("Armazenar dados do teste", key=f"salvar_teste_{imagem_escolhida}"):
                try:
                    if salvar_resultado_teste is not None:
                        retorno = salvar_resultado_teste(payload_resultado)
                        st.success("Dados do teste armazenados com sucesso.")
                        if retorno is not None:
                            st.caption(f"Retorno: {retorno}")
                    else:
                        session_state.resultados_testes_granulometria.append(payload_resultado)
                        st.success("Dados do teste armazenados no session_state com sucesso.")
                except Exception as e:
                    st.error(f"Erro ao armazenar os dados do teste: {e}")

        else:
            st.info("Ainda não há resultados numéricos. Clique em **Detectar bolhas** para gerar tabelas e gráficos.")
