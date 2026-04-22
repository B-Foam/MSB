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
# CALIBRAÇÃO DA BARRA DE 1,0 mm
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
        return None, img_annot

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

    return ww, img_annot


# ============================================================
# MÁSCARA DAS REGIÕES FIXAS
# ============================================================
def criar_mascara_regioes_excluidas(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # topo com texto
    mask[0:int(h * 0.05), :] = 255

    # barra de escala / canto inferior esquerdo
    mask[int(h * 0.78):h, 0:int(w * 0.22)] = 255

    return mask


def aplicar_mascara_exclusao(img_gray, mask_exclusao, valor=255):
    out = img_gray.copy()
    out[mask_exclusao == 255] = valor
    return out


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def gerar_imagens_base(img_bgr, mask_exclusao, clip_limit, bilateral_d, sigma_color, sigma_space):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = aplicar_mascara_exclusao(gray, mask_exclusao, valor=255)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    bilateral = cv2.bilateralFilter(
        clahe_img,
        d=bilateral_d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    return gray, clahe_img, bilateral


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
# CONTORNOS
# ============================================================
def extrair_contornos_candidatos(mask, area_min, area_max, circularidade_min, aspect_tol):
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

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue

        aspect = w / h
        if aspect < (1.0 / aspect_tol) or aspect > aspect_tol:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)

        candidatos.append({
            "contorno": cnt,
            "area": float(area),
            "circularidade": float(circularidade),
            "bbox": (x, y, w, h),
            "cx": float(cx),
            "cy": float(cy),
            "r": float(r)
        })

    return candidatos


def remover_contornos_sobrepostos(candidatos, overlap_factor=0.65):
    if not candidatos:
        return []

    candidatos = sorted(
        candidatos,
        key=lambda c: (c["circularidade"], c["area"]),
        reverse=True
    )

    finais = []

    for c in candidatos:
        manter = True

        for f in finais:
            dist = ((c["cx"] - f["cx"]) ** 2 + (c["cy"] - f["cy"]) ** 2) ** 0.5

            if dist < overlap_factor * min(c["r"], f["r"]):
                manter = False
                break

            if dist < 5 and abs(c["r"] - f["r"]) < 5:
                manter = False
                break

        if manter:
            finais.append(c)

    return finais


def filtrar_borda_contornos(candidatos, shape):
    h, w = shape[:2]
    finais = []

    for c in candidatos:
        x, y, ww, hh = c["bbox"]
        if x <= 2 or y <= 2 or (x + ww) >= w - 2 or (y + hh) >= h - 2:
            continue
        finais.append(c)

    return finais


def desenhar_contornos(img_bgr, candidatos, cor=(0, 255, 0)):
    out = img_bgr.copy()

    for c in candidatos:
        cnt = c["contorno"]
        cv2.drawContours(out, [cnt], -1, cor, 1)

    cv2.putText(
        out,
        f"Contornos finais: {len(candidatos)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        cor,
        2,
        cv2.LINE_AA
    )

    return out


def montar_resumo(candidatos):
    if not candidatos:
        return pd.DataFrame([{
            "Contornos finais": 0,
            "Area media": 0.0,
            "Circularidade media": 0.0,
            "Raio medio": 0.0
        }])

    areas = [c["area"] for c in candidatos]
    circs = [c["circularidade"] for c in candidatos]
    raios = [c["r"] for c in candidatos]

    return pd.DataFrame([{
        "Contornos finais": len(candidatos),
        "Area media": round(float(np.mean(areas)), 2),
        "Circularidade media": round(float(np.mean(circs)), 4),
        "Raio medio": round(float(np.mean(raios)), 2)
    }])


# ============================================================
# TELA PRINCIPAL
# ============================================================
def render_consulta_imagens(
    listar_imagens_supabase,
    montar_url_publica,
    session_state
):
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

        escolhido = st.selectbox(
            "Selecione a imagem",
            lista,
            key="select_imagem_consulta"
        )

        if not escolhido:
            return

        url_imagem = montar_url_publica(escolhido)

        st.markdown("## Configurações do pré-processamento")

        c1, c2, c3 = st.columns(3)
        with c1:
            clip_limit = st.slider("CLAHE - contraste local", 1.0, 5.0, 2.0, 0.1)
            bilateral_d = st.slider("Bilateral - diâmetro", 3, 15, 9, 2)
        with c2:
            sigma_color = st.slider("Bilateral - sigma cor", 10, 150, 75, 5)
            sigma_space = st.slider("Bilateral - sigma espaço", 10, 150, 75, 5)
        with c3:
            kernel_grad = st.slider("Gradiente morfológico - kernel", 3, 21, 5, 2)
            block_size = st.slider("Threshold adaptativo - bloco", 11, 151, 51, 2)

        st.markdown("## Configurações de limpeza dos contornos")

        c4, c5, c6 = st.columns(3)
        with c4:
            c_value = st.slider("Threshold adaptativo - C", 0, 15, 3, 1)
            open_iter = st.slider("Abertura morfológica", 0, 3, 1, 1)
        with c5:
            close_iter = st.slider("Fechamento morfológico", 0, 3, 1, 1)
            area_min = st.slider("Área mínima do contorno (px²)", 5, 5000, 60, 5)
        with c6:
            area_max = st.slider("Área máxima do contorno (px²)", 100, 50000, 4000, 50)
            circularidade_min = st.slider("Circularidade mínima", 0.05, 1.00, 0.35, 0.01)

        c7, c8 = st.columns(2)
        with c7:
            aspect_tol = st.slider("Tolerância largura/altura", 1.0, 3.0, 1.8, 0.1)
        with c8:
            overlap_factor = st.slider("Remoção de sobreposição", 0.2, 1.0, 0.65, 0.05)

        try:
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)

            # calibração
            barra_px_auto, img_calibracao = detectar_barra_escala_px(img_original_bgr)

            st.markdown("## Calibração")
            if barra_px_auto is not None:
                st.success(f"Barra de 1,0 mm detectada automaticamente: {barra_px_auto} px")
            else:
                st.warning("Barra não detectada automaticamente.")

            st.image(
                cv_to_pil(img_calibracao),
                caption="Imagem de calibração com a barra de 1,0 mm marcada em verde",
                use_container_width=True
            )

            # máscara fixa
            mask_exclusao = criar_mascara_regioes_excluidas(img_original_bgr.shape)

            # imagens base
            gray, clahe_img, bilateral = gerar_imagens_base(
                img_original_bgr,
                mask_exclusao,
                clip_limit,
                bilateral_d,
                sigma_color,
                sigma_space
            )

            # gradiente e máscara
            grad = gerar_gradiente_morfologico(bilateral, kernel_grad)
            mask_grad = binarizar_gradiente(
                grad,
                mask_exclusao,
                block_size,
                c_value,
                open_iter,
                close_iter
            )

            # contornos
            contornos_brutos = extrair_contornos_candidatos(
                mask_grad,
                area_min=area_min,
                area_max=area_max,
                circularidade_min=circularidade_min,
                aspect_tol=aspect_tol
            )

            contornos_sem_borda = filtrar_borda_contornos(
                contornos_brutos,
                img_original_bgr.shape
            )

            contornos_finais = remover_contornos_sobrepostos(
                contornos_sem_borda,
                overlap_factor=overlap_factor
            )

            img_contornos = desenhar_contornos(
                img_original_bgr,
                contornos_finais,
                cor=(0, 255, 0)
            )

            st.markdown("## Tratamento inicial")

            p1, p2, p3 = st.columns(3)
            with p1:
                st.caption("Imagem original")
                st.image(img_original_pil, use_container_width=True)
            with p2:
                st.caption("CLAHE")
                st.image(clahe_img, use_container_width=True)
            with p3:
                st.caption("Filtro bilateral")
                st.image(bilateral, use_container_width=True)

            st.markdown("## Diagnóstico do gradiente")

            p4, p5 = st.columns(2)
            with p4:
                st.caption("Gradiente morfológico")
                st.image(grad, use_container_width=True)
            with p5:
                st.caption("Máscara do gradiente")
                st.image(mask_grad, use_container_width=True)

            st.markdown("## Contornos finais sobre a imagem original")
            st.image(cv_to_pil(img_contornos), use_container_width=True)

            st.markdown("## Diagnóstico")
            d1, d2, d3 = st.columns(3)
            d1.metric("Contornos brutos", len(contornos_brutos))
            d2.metric("Após remover borda", len(contornos_sem_borda))
            d3.metric("Contornos finais", len(contornos_finais))

            st.markdown("## Resumo")
            st.dataframe(montar_resumo(contornos_finais), use_container_width=True)

            st.info(
                "Objetivo desta etapa: testar se o gradiente morfológico, com "
                "limpeza mais forte dos contornos, acompanha melhor o anel principal "
                "das bolhas antes de voltar para a medição."
            )

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
