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
# PIPELINE: CLAHE -> BILATERAL -> BLACK-HAT -> ADAPTIVE THRESH
# ============================================================
def preprocessar_pipeline_blackhat(
    img_bgr,
    mask_exclusao,
    clip_limit,
    bilateral_d,
    bilateral_sigma_color,
    bilateral_sigma_space,
    kernel_blackhat,
    block_size,
    c_value,
    usar_reforco
):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = aplicar_mascara_exclusao(gray, mask_exclusao, valor=255)

    # 1) CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )
    clahe_img = clahe.apply(gray)

    # 2) Bilateral
    bilateral = cv2.bilateralFilter(
        clahe_img,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space
    )

    # 3) Black-Hat
    kernel_bh = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_blackhat, kernel_blackhat)
    )
    blackhat = cv2.morphologyEx(bilateral, cv2.MORPH_BLACKHAT, kernel_bh)
    blackhat_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    # 4) Reforço opcional
    if usar_reforco:
        imagem_para_limiar = cv2.addWeighted(bilateral, 1.0, blackhat_norm, 2.0, 0)
        imagem_para_limiar = cv2.normalize(imagem_para_limiar, None, 0, 255, cv2.NORM_MINMAX)
    else:
        imagem_para_limiar = blackhat_norm

    # blockSize precisa ser ímpar
    if block_size % 2 == 0:
        block_size += 1

    # 5) Threshold adaptativo
    th = cv2.adaptiveThreshold(
        imagem_para_limiar,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value
    )

    # inverter para deixar o contorno candidato em branco
    th_inv = cv2.bitwise_not(th)

    # remover regiões excluídas
    th_inv[mask_exclusao == 255] = 0

    # 6) Limpeza leve
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return {
        "gray": gray,
        "clahe": clahe_img,
        "bilateral": bilateral,
        "blackhat": blackhat,
        "blackhat_norm": blackhat_norm,
        "realcado": imagem_para_limiar,
        "threshold": th,
        "mask": mask
    }


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


def desenhar_contornos(img_bgr, candidatos):
    out = img_bgr.copy()

    for c in candidatos:
        cnt = c["contorno"]
        cv2.drawContours(out, [cnt], -1, (0, 255, 0), 1)

    cv2.putText(
        out,
        f"Contornos candidatos: {len(candidatos)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return out


def montar_resumo_contornos(candidatos):
    if not candidatos:
        return pd.DataFrame(columns=["Contornos", "Area media", "Circularidade media"])

    areas = [c["area"] for c in candidatos]
    circ = [c["circularidade"] for c in candidatos]

    return pd.DataFrame([{
        "Contornos": len(candidatos),
        "Area media": round(float(np.mean(areas)), 2),
        "Circularidade media": round(float(np.mean(circ)), 4)
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

        st.markdown("## Configurações do pipeline")

        col1, col2, col3 = st.columns(3)
        with col1:
            clip_limit = st.slider("CLAHE - contraste local", 1.0, 5.0, 2.0, 0.1)
            bilateral_d = st.slider("Bilateral - diâmetro", 3, 15, 9, 2)
            bilateral_sigma_color = st.slider("Bilateral - sigma cor", 10, 150, 75, 5)

        with col2:
            bilateral_sigma_space = st.slider("Bilateral - sigma espaço", 10, 150, 75, 5)
            kernel_blackhat = st.slider("Black-Hat - kernel", 5, 41, 15, 2)
            if kernel_blackhat % 2 == 0:
                kernel_blackhat += 1
            usar_reforco = st.checkbox("Usar reforço do Black-Hat", value=True)

        with col3:
            block_size = st.slider("Threshold - bloco local", 11, 151, 51, 2)
            if block_size % 2 == 0:
                block_size += 1
            c_value = st.slider("Threshold - ajuste fino (C)", 0, 15, 3, 1)

        st.markdown("## Configurações dos contornos")

        col4, col5, col6 = st.columns(3)
        with col4:
            area_min = st.slider("Área mínima do contorno (px²)", 5, 5000, 40, 5)
        with col5:
            area_max = st.slider("Área máxima do contorno (px²)", 100, 50000, 5000, 50)
        with col6:
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

            # preprocessamento
            mask_exclusao = criar_mascara_regioes_excluidas(img_original_bgr.shape)

            resultado = preprocessar_pipeline_blackhat(
                img_bgr=img_original_bgr,
                mask_exclusao=mask_exclusao,
                clip_limit=clip_limit,
                bilateral_d=bilateral_d,
                bilateral_sigma_color=bilateral_sigma_color,
                bilateral_sigma_space=bilateral_sigma_space,
                kernel_blackhat=kernel_blackhat,
                block_size=block_size,
                c_value=c_value,
                usar_reforco=usar_reforco
            )

            st.markdown("## Diagnóstico do pré-processamento")

            p1, p2, p3 = st.columns(3)
            with p1:
                st.caption("Imagem original")
                st.image(img_original_pil, use_container_width=True)
            with p2:
                st.caption("CLAHE")
                st.image(resultado["clahe"], use_container_width=True)
            with p3:
                st.caption("Filtro bilateral")
                st.image(resultado["bilateral"], use_container_width=True)

            p4, p5, p6 = st.columns(3)
            with p4:
                st.caption("Black-Hat")
                st.image(resultado["blackhat"], use_container_width=True)
            with p5:
                st.caption("Black-Hat normalizado")
                st.image(resultado["blackhat_norm"], use_container_width=True)
            with p6:
                st.caption("Imagem reforçada")
                st.image(resultado["realcado"], use_container_width=True)

            p7, p8 = st.columns(2)
            with p7:
                st.caption("Threshold adaptativo")
                st.image(resultado["threshold"], use_container_width=True)
            with p8:
                st.caption("Máscara final")
                st.image(resultado["mask"], use_container_width=True)

            # contornos
            candidatos = extrair_contornos_candidatos(
                resultado["mask"],
                area_min=area_min,
                area_max=area_max,
                circularidade_min=circularidade_min
            )

            img_contornos = desenhar_contornos(img_original_bgr, candidatos)

            st.markdown("## Diagnóstico dos contornos")

            cta, ctb = st.columns(2)
            with cta:
                st.caption(f"Máscara final — {len(candidatos)} contornos")
                st.image(resultado["mask"], use_container_width=True)
            with ctb:
                st.caption("Contornos sobre a imagem original")
                st.image(cv_to_pil(img_contornos), use_container_width=True)

            st.markdown("## Resumo")
            st.dataframe(montar_resumo_contornos(candidatos), use_container_width=True)

            st.info(
                "Esta etapa serve para validar visualmente se o pipeline "
                "CLAHE → Bilateral → Black-Hat → Adaptive Threshold está "
                "gerando contornos candidatos próximos das bordas reais das bolhas."
            )

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
