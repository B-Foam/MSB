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
# PRÉ-PROCESSAMENTO PRINCIPAL
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


def gerar_dog_suave(img_gray, sigma1, sigma2):
    g1 = cv2.GaussianBlur(img_gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(img_gray, (0, 0), sigma2)
    dog = cv2.subtract(g2, g1)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    return dog


def binarizar_adaptativo(img_gray, block_size, c_value, mask_exclusao):
    if block_size % 2 == 0:
        block_size += 1

    th = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value
    )

    th[mask_exclusao == 255] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    return th


# ============================================================
# CONTORNOS CANDIDATOS
# ============================================================
def extrair_contornos_candidatos(mask, area_min, area_max, circularidade_min):
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

        candidatos.append({
            "contorno": cnt,
            "area": float(area),
            "circularidade": float(circularidade),
            "bbox": (x, y, w, h)
        })

    return candidatos


def desenhar_contornos(img_bgr, candidatos, cor=(0, 255, 0)):
    out = img_bgr.copy()

    for c in candidatos:
        cnt = c["contorno"]
        cv2.drawContours(out, [cnt], -1, cor, 1)

    cv2.putText(
        out,
        f"Contornos candidatos: {len(candidatos)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        cor,
        2,
        cv2.LINE_AA
    )

    return out


def montar_resumo(nome, candidatos):
    if not candidatos:
        return {
            "Filtro": nome,
            "Contornos": 0,
            "Area media": 0.0,
            "Circularidade media": 0.0,
        }

    areas = [c["area"] for c in candidatos]
    circs = [c["circularidade"] for c in candidatos]

    return {
        "Filtro": nome,
        "Contornos": len(candidatos),
        "Area media": round(float(np.mean(areas)), 2),
        "Circularidade media": round(float(np.mean(circs)), 4),
    }


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

        col1, col2, col3 = st.columns(3)
        with col1:
            clip_limit = st.slider("CLAHE - contraste local", 1.0, 5.0, 2.0, 0.1)
            bilateral_d = st.slider("Bilateral - diâmetro", 3, 15, 9, 2)
        with col2:
            sigma_color = st.slider("Bilateral - sigma cor", 10, 150, 75, 5)
            sigma_space = st.slider("Bilateral - sigma espaço", 10, 150, 75, 5)
        with col3:
            kernel_grad = st.slider("Gradiente morfológico - kernel", 3, 21, 5, 2)
            block_size = st.slider("Threshold adaptativo - bloco", 11, 151, 51, 2)

        st.markdown("## Configurações do DoG e contornos")

        col4, col5, col6 = st.columns(3)
        with col4:
            sigma1 = st.slider("DoG - sigma 1", 0.5, 5.0, 1.0, 0.1)
            sigma2 = st.slider("DoG - sigma 2", 1.0, 8.0, 2.0, 0.1)
        with col5:
            c_value = st.slider("Threshold adaptativo - C", 0, 15, 3, 1)
            area_min = st.slider("Área mínima do contorno (px²)", 5, 5000, 40, 5)
        with col6:
            area_max = st.slider("Área máxima do contorno (px²)", 100, 50000, 5000, 50)
            circularidade_min = st.slider("Circularidade mínima", 0.05, 1.00, 0.20, 0.01)

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

            # filtros principais
            grad = gerar_gradiente_morfologico(bilateral, kernel_grad)
            dog = gerar_dog_suave(bilateral, sigma1, sigma2)

            # máscaras binárias
            mask_grad = binarizar_adaptativo(grad, block_size, c_value, mask_exclusao)
            mask_dog = binarizar_adaptativo(dog, block_size, c_value, mask_exclusao)

            # contornos
            cont_grad = extrair_contornos_candidatos(
                mask_grad,
                area_min=area_min,
                area_max=area_max,
                circularidade_min=circularidade_min
            )

            cont_dog = extrair_contornos_candidatos(
                mask_dog,
                area_min=area_min,
                area_max=area_max,
                circularidade_min=circularidade_min
            )

            img_cont_grad = desenhar_contornos(img_original_bgr, cont_grad, cor=(0, 255, 0))
            img_cont_dog = desenhar_contornos(img_original_bgr, cont_dog, cor=(255, 0, 0))

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

            p4, p5 = st.columns(2)
            with p4:
                st.caption("Gradiente morfológico")
                st.image(grad, use_container_width=True)
            with p5:
                st.caption("DoG suave")
                st.image(dog, use_container_width=True)

            st.markdown("## Máscaras candidatas")

            m1, m2 = st.columns(2)
            with m1:
                st.caption(f"Máscara do gradiente — {len(cont_grad)} contornos")
                st.image(mask_grad, use_container_width=True)
            with m2:
                st.caption(f"Máscara do DoG — {len(cont_dog)} contornos")
                st.image(mask_dog, use_container_width=True)

            st.markdown("## Contornos sobre a imagem original")

            cta, ctb = st.columns(2)
            with cta:
                st.caption("Contornos do gradiente")
                st.image(cv_to_pil(img_cont_grad), use_container_width=True)
            with ctb:
                st.caption("Contornos do DoG")
                st.image(cv_to_pil(img_cont_dog), use_container_width=True)

            resumo = pd.DataFrame([
                montar_resumo("Gradiente morfológico", cont_grad),
                montar_resumo("DoG suave", cont_dog),
            ])

            st.markdown("## Resumo")
            st.dataframe(resumo, use_container_width=True)

            st.info(
                "Objetivo desta etapa: descobrir se o gradiente morfológico "
                "ou o DoG suave gera contornos mais próximos das bordas reais "
                "das bolhas, antes de voltar para a medição."
            )

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
