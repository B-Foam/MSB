import io
import requests
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image


# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================
def baixar_imagem(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv):
    if len(img_cv.shape) == 2:
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ============================================================
# CALIBRAÇÃO DA BARRA
# ============================================================
def detectar_barra_escala_px(img_bgr):
    img_annot = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    x0 = 0
    x1 = int(w * 0.35)
    y0 = int(h * 0.78)
    y1 = h

    crop = img_bgr[y0:y1, x0:x1].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor = None
    melhor_area = 0

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / max(hh, 1)
        area = ww * hh

        if ww > 60 and hh < 20 and aspect > 5:
            if area > melhor_area:
                melhor_area = area
                melhor = (x, y, ww, hh)

    if melhor is None:
        return None, img_annot, None

    x, y, ww, hh = melhor
    gx = x0 + x
    gy = y0 + y

    cv2.rectangle(img_annot, (gx, gy), (gx + ww, gy + hh), (0, 255, 0), 2)
    cv2.line(
        img_annot,
        (gx, gy + hh // 2),
        (gx + ww, gy + hh // 2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        img_annot,
        f"1.0 mm = {ww} px",
        (gx, max(20, gy - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    barra_info = {"x": gx, "y": gy, "w": ww, "h": hh}
    return ww, img_annot, barra_info


# ============================================================
# REGIÕES EXCLUÍDAS
# ============================================================
def criar_mascara_regioes_excluidas(shape, barra_info=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # faixa superior do texto
    mask[0:int(h * 0.035), :] = 255

    # régua
    if barra_info is not None:
        x = barra_info["x"]
        y = barra_info["y"]
        ww = barra_info["w"]
        hh = barra_info["h"]

        x_ini = max(0, x - 25)
        y_ini = max(0, y - 60)
        x_fim = min(w, x + ww + 90)
        y_fim = min(h, y + hh + 20)

        mask[y_ini:y_fim, x_ini:x_fim] = 255

    return mask


def aplicar_mascara_exclusao(img_gray, mask_exclusao, valor=255):
    out = img_gray.copy()
    out[mask_exclusao == 255] = valor
    return out


# ============================================================
# PARÂMETROS AUTOMÁTICOS
# ============================================================
def obter_parametros_automaticos(shape):
    h, w = shape[:2]
    base = min(h, w)

    params = {
        "clip_limit": 2.0,
        "bilateral_d": 9,
        "sigma_color": 75,
        "sigma_space": 75,
        "kernel_grad": 5,
        "block_size": 41,
        "c_value": 2,
        "open_iter": 0,
        "close_iter": 1,
        "area_min": max(40, int(base * 0.015)),
        "area_max": int(base * 25),
        "circularidade_min": 0.28,
        "solidez_min": 0.55,
        "raio_min": max(6, int(base * 0.0045)),
        "raio_max": max(40, int(base * 0.08)),
        "aspect_tol": 2.2,
        "margem_x_frac": 0.10,
        "margem_y_frac": 0.07,
        "fator_dist": 1.25,
    }

    if params["block_size"] % 2 == 0:
        params["block_size"] += 1

    return params


# ============================================================
# PIPELINE
# ============================================================
def gerar_imagens_base(img_bgr, mask_exclusao, clip_limit, bilateral_d, sigma_color, sigma_space):
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_full = clahe.apply(gray_full)

    bilateral_full = cv2.bilateralFilter(
        clahe_full,
        d=bilateral_d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    gray_vis = gray_full.copy()
    clahe_vis = clahe_full.copy()
    bilateral_vis = bilateral_full.copy()

    gray_proc = aplicar_mascara_exclusao(gray_full, mask_exclusao, valor=255)
    clahe_proc = aplicar_mascara_exclusao(clahe_full, mask_exclusao, valor=255)
    bilateral_proc = aplicar_mascara_exclusao(bilateral_full, mask_exclusao, valor=255)

    return gray_vis, clahe_vis, bilateral_vis, gray_proc, clahe_proc, bilateral_proc


def gerar_gradiente_morfologico(img_gray, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    grad = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX)
    return grad


def binarizar_gradiente(grad, mask_exclusao, block_size, c_value, open_iter, close_iter):
    if block_size % 2 == 0:
        block_size += 1

    th = cv2.adaptiveThreshold(
        grad,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value
    )

    th[mask_exclusao == 255] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if open_iter > 0:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    if close_iter > 0:
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return th


# ============================================================
# FILTRO ESPACIAL
# ============================================================
def dentro_da_regiao_util(cx, cy, shape, margem_x_frac, margem_y_frac):
    h, w = shape[:2]
    mx = int(w * margem_x_frac)
    my = int(h * margem_y_frac)
    return (mx <= cx <= (w - mx)) and (my <= cy <= (h - my))


# ============================================================
# CANDIDATOS
# ============================================================
def extrair_candidatos_circulares(
    mask,
    area_min,
    area_max,
    circularidade_min,
    solidez_min,
    raio_min,
    raio_max,
    aspect_tol,
    shape_img,
    margem_x_frac,
    margem_y_frac
):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_min or area > area_max:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        circularidade = 4 * np.pi * area / (peri * peri)
        if circularidade < circularidade_min:
            continue

        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull)
        if area_hull <= 0:
            continue

        solidez = area / area_hull
        if solidez < solidez_min:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue

        aspect = w / h
        if aspect < (1.0 / aspect_tol) or aspect > aspect_tol:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < raio_min or r > raio_max:
            continue

        if not dentro_da_regiao_util(cx, cy, shape_img, margem_x_frac, margem_y_frac):
            continue

        candidatos.append({
            "contorno": cnt,
            "area": float(area),
            "circularidade": float(circularidade),
            "solidez": float(solidez),
            "bbox": (x, y, w, h),
            "cx": float(cx),
            "cy": float(cy),
            "r": float(r)
        })

    return candidatos


def agrupar_por_proximidade(candidatos, fator_dist=1.0):
    grupos = []
    usados = [False] * len(candidatos)

    for i, c in enumerate(candidatos):
        if usados[i]:
            continue

        grupo = [i]
        usados[i] = True
        mudou = True

        while mudou:
            mudou = False
            for j, cj in enumerate(candidatos):
                if usados[j]:
                    continue

                for idx in grupo:
                    ci = candidatos[idx]
                    dist = ((ci["cx"] - cj["cx"]) ** 2 + (ci["cy"] - cj["cy"]) ** 2) ** 0.5
                    limite = fator_dist * max(ci["r"], cj["r"])
                    if dist <= limite:
                        grupo.append(j)
                        usados[j] = True
                        mudou = True
                        break

        grupos.append([candidatos[idx] for idx in grupo])

    return grupos


def escolher_candidato_dominante(grupo, shape_img):
    h, w = shape_img[:2]
    centro_x = w / 2.0
    centro_y = h / 2.0
    diag = (w ** 2 + h ** 2) ** 0.5

    melhor = None
    melhor_score = -1e9

    for c in grupo:
        dist_centro = ((c["cx"] - centro_x) ** 2 + (c["cy"] - centro_y) ** 2) ** 0.5
        fator_central = 1.0 - (dist_centro / diag)

        score = (
            2.0 * c["circularidade"] +
            1.8 * c["solidez"] +
            0.0015 * c["area"] +
            0.06 * c["r"] +
            1.2 * fator_central
        )

        if score > melhor_score:
            melhor_score = score
            melhor = c

    return melhor


def manter_candidatos_dominantes(candidatos, fator_dist, shape_img):
    if not candidatos:
        return []

    grupos = agrupar_por_proximidade(candidatos, fator_dist=fator_dist)
    finais = [escolher_candidato_dominante(g, shape_img) for g in grupos if len(g) > 0]
    return finais


# ============================================================
# DESENHO
# ============================================================
def desenhar_circulos_coloridos(img_bgr, candidatos):
    out = img_bgr.copy()
    rng = np.random.default_rng(42)

    for c in candidatos:
        color = tuple(int(v) for v in rng.integers(60, 256, size=3))
        cx = int(round(c["cx"]))
        cy = int(round(c["cy"]))
        r = int(round(c["r"]))
        cv2.circle(out, (cx, cy), r, color, 2)

    cv2.putText(
        out,
        f"Candidatos dominantes: {len(candidatos)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return out


def desenhar_regiao_util(img_bgr, margem_x_frac, margem_y_frac):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    mx = int(w * margem_x_frac)
    my = int(h * margem_y_frac)
    cv2.rectangle(out, (mx, my), (w - mx, h - my), (255, 255, 0), 2)
    return out


def montar_resumo(candidatos):
    if not candidatos:
        return pd.DataFrame([{
            "Bolhas detectadas": 0,
            "Area media": 0.0,
            "Circularidade media": 0.0,
            "Solidez media": 0.0,
            "Raio medio": 0.0
        }])

    areas = [c["area"] for c in candidatos]
    circs = [c["circularidade"] for c in candidatos]
    sols = [c["solidez"] for c in candidatos]
    raios = [c["r"] for c in candidatos]

    return pd.DataFrame([{
        "Bolhas detectadas": len(candidatos),
        "Area media": round(float(np.mean(areas)), 2),
        "Circularidade media": round(float(np.mean(circs)), 4),
        "Solidez media": round(float(np.mean(sols)), 4),
        "Raio medio": round(float(np.mean(raios)), 2)
    }])


# ============================================================
# TELA PRINCIPAL
# ============================================================
def render_consulta_imagens(listar_imagens_supabase, montar_url_publica, session_state):
    with st.container(border=True):
        if st.button("Atualizar lista", key="btn_atualizar_lista_consulta"):
            imagens, erro = listar_imagens_supabase("")
            if erro:
                st.error(f"Erro: {erro}")
            else:
                session_state.lista_imagens_consulta = [img["name"] for img in imagens]

        lista = session_state.get("lista_imagens_consulta", [])
        if not lista:
            st.info("Clique em 'Atualizar lista' para carregar as imagens.")
            return

        escolhido = st.selectbox("Selecione a imagem", lista, key="select_imagem_consulta")
        if not escolhido:
            return

        try:
            url_imagem = montar_url_publica(escolhido)
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)

            params = obter_parametros_automaticos(img_original_bgr.shape)

            barra_px_auto, img_calibracao, barra_info = detectar_barra_escala_px(img_original_bgr)
            mask_exclusao = criar_mascara_regioes_excluidas(img_original_bgr.shape, barra_info)

            gray_vis, clahe_vis, bilateral_vis, _, _, bilateral_proc = gerar_imagens_base(
                img_original_bgr,
                mask_exclusao,
                params["clip_limit"],
                params["bilateral_d"],
                params["sigma_color"],
                params["sigma_space"]
            )

            grad_vis = gerar_gradiente_morfologico(bilateral_vis, params["kernel_grad"])
            grad_proc = gerar_gradiente_morfologico(bilateral_proc, params["kernel_grad"])

            mask_grad = binarizar_gradiente(
                grad_proc,
                mask_exclusao,
                params["block_size"],
                params["c_value"],
                params["open_iter"],
                params["close_iter"]
            )

            candidatos_brutos = extrair_candidatos_circulares(
                mask_grad,
                params["area_min"],
                params["area_max"],
                params["circularidade_min"],
                params["solidez_min"],
                params["raio_min"],
                params["raio_max"],
                params["aspect_tol"],
                img_original_bgr.shape,
                params["margem_x_frac"],
                params["margem_y_frac"]
            )

            candidatos_finais = manter_candidatos_dominantes(
                candidatos_brutos,
                params["fator_dist"],
                img_original_bgr.shape
            )

            img_regiao = desenhar_regiao_util(
                img_original_bgr,
                params["margem_x_frac"],
                params["margem_y_frac"]
            )

            img_candidatos = desenhar_circulos_coloridos(
                img_original_bgr,
                candidatos_finais
            )

            st.markdown("## Calibração")
            if barra_px_auto is not None:
                st.success(f"Barra de 1,0 mm detectada automaticamente: {barra_px_auto} px")
            else:
                st.warning("Barra não detectada automaticamente.")

            st.image(cv_to_pil(img_calibracao), caption="Barra de 1,0 mm detectada", width=480)

            st.markdown("## Pré-processamento automático")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("Imagem original")
                st.image(img_original_pil, width=250)
            with c2:
                st.caption("CLAHE")
                st.image(clahe_vis, width=250)
            with c3:
                st.caption("Filtro bilateral")
                st.image(bilateral_vis, width=250)

            c4, c5 = st.columns(2)
            with c4:
                st.caption("Gradiente morfológico")
                st.image(grad_vis, width=390)
            with c5:
                st.caption("Máscara automática")
                st.image(mask_grad, width=390)

            st.markdown("## Região útil")
            st.image(cv_to_pil(img_regiao), width=720)

            st.markdown("## Resultado automático")
            st.image(cv_to_pil(img_candidatos), width=720)

            st.markdown("## Diagnóstico")
            d1, d2 = st.columns(2)
            d1.metric("Candidatos brutos", len(candidatos_brutos))
            d2.metric("Bolhas detectadas", len(candidatos_finais))

            st.markdown("## Resumo")
            st.dataframe(montar_resumo(candidatos_finais), use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
