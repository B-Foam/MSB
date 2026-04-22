import io
import requests
import streamlit as st
import numpy as np
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
# MÁSCARAS FIXAS
# ============================================================
def criar_mascara_regioes_excluidas(shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Faixa superior com texto
    mask[0:int(h * 0.05), :] = 255

    # Barra de escala e canto inferior esquerdo
    mask[int(h * 0.78):h, 0:int(w * 0.22)] = 255

    return mask


def aplicar_mascara_exclusao(img_gray, mask_exclusao):
    out = img_gray.copy()
    out[mask_exclusao == 255] = 255
    return out


# ============================================================
# FILTROS DE DIAGNÓSTICO
# ============================================================
def gerar_filtros_diagnostico(img_bgr, mask_exclusao):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = aplicar_mascara_exclusao(gray, mask_exclusao)

    filtros = {}

    filtros["Cinza"] = gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    filtros["CLAHE"] = clahe

    bilateral = cv2.bilateralFilter(clahe, 9, 75, 75)
    filtros["Bilateral"] = bilateral

    mediana = cv2.medianBlur(clahe, 5)
    filtros["Mediana"] = mediana

    gauss = cv2.GaussianBlur(clahe, (5, 5), 0)
    filtros["Gaussiano"] = gauss

    # Gradiente morfológico
    kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grad_morf = cv2.morphologyEx(bilateral, cv2.MORPH_GRADIENT, kernel_grad)
    filtros["Gradiente morfológico"] = grad_morf

    # Laplaciano
    lap = cv2.Laplacian(gauss, cv2.CV_8U, ksize=3)
    filtros["Laplaciano"] = lap

    # DoG suave
    g1 = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(clahe, (0, 0), 2.0)
    dog = cv2.subtract(g2, g1)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    filtros["DoG suave"] = dog

    # Canny fraco
    canny = cv2.Canny(gauss, 15, 60)
    filtros["Canny fraco"] = canny

    # Canny médio
    canny2 = cv2.Canny(gauss, 30, 90)
    filtros["Canny médio"] = canny2

    # Threshold adaptativo 1
    th1 = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        2
    )
    th1[mask_exclusao == 255] = 0
    filtros["Adaptativo 31x31 C=2"] = th1

    # Threshold adaptativo 2
    th2 = cv2.adaptiveThreshold(
        bilateral,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        4
    )
    th2[mask_exclusao == 255] = 0
    filtros["Adaptativo 51x51 C=4"] = th2

    # Otsu
    _, otsu = cv2.threshold(
        bilateral,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    otsu[mask_exclusao == 255] = 0
    filtros["Otsu"] = otsu

    return filtros


# ============================================================
# CONTORNOS
# ============================================================
def extrair_contornos_candidatos(
    img_gray,
    area_min,
    area_max,
    circularidade_min
):
    # Garantir binário
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)

    # Se não for binário, binariza com Otsu
    valores_unicos = np.unique(img_gray)
    if len(valores_unicos) > 4:
        _, bin_img = cv2.threshold(
            img_gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        bin_img = img_gray.copy()

    # Se maioria do objeto estiver preta, inverter
    if np.mean(bin_img) > 127:
        bin_img = cv2.bitwise_not(bin_img)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            "area": area,
            "circularidade": circularidade,
            "bbox": (x, y, w, h)
        })

    return candidatos, bin_img


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


# ============================================================
# PIPELINE PRINCIPAL DE DIAGNÓSTICO
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

        st.markdown("## Configurações do diagnóstico")

        c1, c2, c3 = st.columns(3)
        with c1:
            area_min = st.slider("Área mínima do contorno (px²)", 5, 5000, 40, 5)
        with c2:
            area_max = st.slider("Área máxima do contorno (px²)", 100, 50000, 5000, 50)
        with c3:
            circularidade_min = st.slider("Circularidade mínima", 0.05, 1.00, 0.20, 0.01)

        try:
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)

            # Calibração
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

            # Máscara fixa
            mask_exclusao = criar_mascara_regioes_excluidas(img_original_bgr.shape)

            # Filtros
            filtros = gerar_filtros_diagnostico(img_original_bgr, mask_exclusao)

            st.markdown("## Diagnóstico dos filtros")

            nomes = list(filtros.keys())

            # Mostrar filtros em grade 3 colunas
            for i in range(0, len(nomes), 3):
                cols = st.columns(3)
                for j in range(3):
                    idx = i + j
                    if idx < len(nomes):
                        nome = nomes[idx]
                        with cols[j]:
                            st.caption(nome)
                            st.image(filtros[nome], use_container_width=True)

            st.markdown("## Diagnóstico dos contornos")

            resultados = []

            for nome, img_filtro in filtros.items():
                candidatos, bin_img = extrair_contornos_candidatos(
                    img_filtro,
                    area_min=area_min,
                    area_max=area_max,
                    circularidade_min=circularidade_min
                )

                img_contornos = desenhar_contornos(img_original_bgr, candidatos)

                resultados.append({
                    "nome": nome,
                    "binaria": bin_img,
                    "candidatos": candidatos,
                    "imagem_contornos": img_contornos,
                    "quantidade": len(candidatos)
                })

            # Ordenar por quantidade crescente só para facilitar leitura
            resultados = sorted(resultados, key=lambda x: x["quantidade"])

            for r in resultados:
                st.markdown(f"### {r['nome']}")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption(f"Máscara / binário — {r['quantidade']} contornos")
                    st.image(r["binaria"], use_container_width=True)
                with col_b:
                    st.caption("Contornos sobre a imagem original")
                    st.image(cv_to_pil(r["imagem_contornos"]), use_container_width=True)

            resumo = [{"Filtro": r["nome"], "Quantidade de contornos": r["quantidade"]} for r in resultados]
            st.markdown("## Resumo")
            st.dataframe(resumo, use_container_width=True)

            st.info(
                "Objetivo desta etapa: identificar visualmente qual filtro gera "
                "contornos mais próximos das bordas reais das bolhas, antes de "
                "voltar para a segmentação e medição."
            )

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
