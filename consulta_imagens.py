import io
import requests
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
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
# CALIBRAÇÃO PELA BARRA DE 1,0 mm
# ============================================================
def detectar_barra_escala_px(img_bgr):
    """
    Procura a barra preta horizontal no canto inferior esquerdo.
    Retorna:
        - comprimento em pixels
        - imagem anotada com a barra desenhada em verde
    """
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
def mascarar_regioes_fixas(img_gray):
    out = img_gray.copy()
    h, w = out.shape[:2]

    # Texto do topo
    out[0:int(h * 0.05), :] = 255

    # Região da barra de escala
    out[int(h * 0.78):h, 0:int(w * 0.22)] = 255

    return out


# ============================================================
# FILTROS DE PRÉ-PROCESSAMENTO
# ============================================================
def filtro_clahe(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


def filtro_correcao_fundo(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 35)
    corr = cv2.addWeighted(gray, 1.8, background, -0.8, 0)
    corr = cv2.normalize(corr, None, 0, 255, cv2.NORM_MINMAX)
    return corr


def filtro_dog(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(gray, (0, 0), 2.2)
    dog = cv2.subtract(g2, g1)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dog = clahe.apply(dog)
    return dog


def filtro_canny(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 15, 60)
    return edges


def filtro_adaptativo(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return th


def filtro_gradiente(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return grad


# ============================================================
# DETECÇÃO POR HOUGH
# ============================================================
def detectar_bolhas_hough(
    img_gray,
    sensibilidade_busca,
    distancia_minima,
    forca_borda,
    rigor_confirmacao,
    raio_minimo,
    raio_maximo
):
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=sensibilidade_busca,
        minDist=distancia_minima,
        param1=forca_borda,
        param2=rigor_confirmacao,
        minRadius=raio_minimo,
        maxRadius=raio_maximo
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int)
    return circles.tolist()


# ============================================================
# DETECÇÃO POR CONTORNO
# ============================================================
def detectar_bolhas_contorno(bin_img, raio_minimo, raio_maximo):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circulos = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circularidade = 4 * np.pi * area / (peri * peri)
        (x, y), r = cv2.minEnclosingCircle(cnt)

        if r < raio_minimo or r > raio_maximo:
            continue

        if circularidade >= 0.35:
            circulos.append([int(x), int(y), int(r)])

    return circulos


# ============================================================
# DETECÇÃO POR WATERSHED
# ============================================================
def detectar_bolhas_watershed(img_gray, raio_minimo, raio_maximo):
    """
    Usa threshold + distance transform + watershed para separar
    regiões coladas e gerar candidatos circulares.
    """
    # threshold
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # inverter se necessário, queremos regiões candidatas claras
    media_obj = float(np.mean(img_gray[th == 255])) if np.any(th == 255) else 0
    media_bg = float(np.mean(img_gray[th == 0])) if np.any(th == 0) else 0
    if media_obj < media_bg:
        th = cv2.bitwise_not(th)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if np.max(dist) <= 0:
        return []

    _, sure_fg = cv2.threshold(dist, 0.30 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    circulos = []

    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue

        mask = np.uint8(markers == marker_id) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue

            circularidade = 4 * np.pi * area / (peri * peri)
            (x, y), r = cv2.minEnclosingCircle(cnt)

            if r < raio_minimo or r > raio_maximo:
                continue

            if circularidade >= 0.25:
                circulos.append([int(x), int(y), int(r)])

    return circulos


# ============================================================
# LIMPEZA DE DETECÇÕES
# ============================================================
def remover_circulos_duplicados(circulos, fator=0.72):
    if not circulos:
        return []

    circulos = sorted(circulos, key=lambda c: c[2], reverse=True)
    finais = []

    for x, y, r in circulos:
        manter = True

        for xf, yf, rf in finais:
            dist = ((x - xf) ** 2 + (y - yf) ** 2) ** 0.5

            if dist < fator * min(r, rf):
                manter = False
                break

            if dist < 6 and abs(r - rf) < 8:
                manter = False
                break

        if manter:
            finais.append([x, y, r])

    return finais


def filtrar_circulos_borda(circulos, shape):
    h, w = shape[:2]
    finais = []

    for x, y, r in circulos:
        if x - r <= 4 or y - r <= 4 or x + r >= w - 4 or y + r >= h - 4:
            continue
        finais.append([x, y, r])

    return finais


def remover_circulos_muito_grandes(circulos, img_shape):
    h, w = img_shape[:2]
    limite = min(h, w) * 0.12

    finais = []
    for x, y, r in circulos:
        if r <= limite:
            finais.append([x, y, r])

    return finais


def filtrar_por_assinatura_radial(img_gray, circulos):
    """
    Mantém círculos cuja borda se comporta como bolha real.
    """
    h, w = img_gray.shape[:2]
    finais = []

    for x, y, r in circulos:
        if r < 5:
            continue

        mask_centro = np.zeros((h, w), dtype=np.uint8)
        mask_anel = np.zeros((h, w), dtype=np.uint8)
        mask_externo = np.zeros((h, w), dtype=np.uint8)

        r_centro = max(2, int(r * 0.42))
        r_anel_in = max(r_centro + 1, int(r * 0.72))
        r_anel_out = int(r * 1.00)
        r_ext_in = int(r * 1.05)
        r_ext_out = int(r * 1.22)

        cv2.circle(mask_centro, (x, y), r_centro, 255, -1)

        cv2.circle(mask_anel, (x, y), r_anel_out, 255, -1)
        cv2.circle(mask_anel, (x, y), r_anel_in, 0, -1)

        cv2.circle(mask_externo, (x, y), r_ext_out, 255, -1)
        cv2.circle(mask_externo, (x, y), r_ext_in, 0, -1)

        vals_centro = img_gray[mask_centro == 255]
        vals_anel = img_gray[mask_anel == 255]
        vals_externo = img_gray[mask_externo == 255]

        if len(vals_centro) < 10 or len(vals_anel) < 10 or len(vals_externo) < 10:
            continue

        media_centro = float(np.mean(vals_centro))
        media_anel = float(np.mean(vals_anel))
        media_externo = float(np.mean(vals_externo))

        contraste_anel_centro = abs(media_anel - media_centro)
        contraste_anel_externo = abs(media_anel - media_externo)

        if contraste_anel_centro >= 4 and contraste_anel_externo >= 3:
            finais.append([x, y, r])

    return finais


# ============================================================
# ANOTAÇÃO / TABELAS / GRÁFICOS
# ============================================================
def desenhar_bolhas(img_bgr, circulos, px_per_mm):
    out = img_bgr.copy()

    for x, y, r in circulos:
        diam_um = (2 * r / px_per_mm) * 1000.0

        cv2.circle(out, (x, y), r, (0, 255, 0), 2)

        if diam_um >= 300:
            cv2.putText(
                out,
                f"{int(diam_um)}",
                (x - 12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

    cv2.putText(
        out,
        f"Bolhas detectadas: {len(circulos)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return out


def montar_dataframe_medidas(circulos, px_per_mm):
    dados = []

    for i, (x, y, r) in enumerate(circulos, start=1):
        diam_um = (2 * r / px_per_mm) * 1000.0
        raio_um = (r / px_per_mm) * 1000.0

        dados.append({
            "Bolha": i,
            "Centro X (px)": x,
            "Centro Y (px)": y,
            "Raio (px)": r,
            "Raio (µm)": round(raio_um, 2),
            "Diâmetro (µm)": round(diam_um, 2)
        })

    return pd.DataFrame(dados)


def plotar_distribuicao(df):
    diametros = df["Diâmetro (µm)"].values

    max_d = max(600, int(np.ceil(diametros.max() / 100.0) * 100))
    bins = list(range(0, max_d + 100, 100))

    contagens, bordas = np.histogram(diametros, bins=bins)
    centros = [(bordas[i] + bordas[i + 1]) / 2 for i in range(len(bordas) - 1)]
    labels = [f"{int(bordas[i])}-{int(bordas[i+1])}" for i in range(len(bordas) - 1)]

    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    ax_bar.bar(labels, contagens)
    ax_bar.set_title("Quantidade de bolhas por faixa de diâmetro")
    ax_bar.set_xlabel("Faixa de diâmetro (µm)")
    ax_bar.set_ylabel("Quantidade")
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig_curve, ax_curve = plt.subplots(figsize=(10, 4))
    ax_curve.plot(centros, contagens, marker="o")
    ax_curve.set_title("Curva de distribuição do tamanho das bolhas")
    ax_curve.set_xlabel("Diâmetro (µm)")
    ax_curve.set_ylabel("Quantidade")
    plt.tight_layout()

    return fig_bar, fig_curve, contagens, labels


# ============================================================
# PIPELINE PRINCIPAL
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

        st.markdown("## Configurações da detecção")

        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            sensibilidade_busca = st.slider(
                "Sensibilidade da busca de círculos",
                1.0, 2.5, 1.2, 0.1,
                help="Valores menores deixam a busca mais detalhada. Valores maiores simplificam a busca."
            )
            distancia_minima = st.slider(
                "Distância mínima entre centros",
                8, 120, 16, 1,
                help="Aumente se estiver marcando duas bolhas quase no mesmo lugar."
            )

        with colp2:
            forca_borda = st.slider(
                "Força mínima da borda",
                20, 200, 55, 1,
                help="Aumente se estiver detectando muito ruído. Diminua se estiver perdendo bolhas."
            )
            rigor_confirmacao = st.slider(
                "Rigor para confirmar a bolha",
                8, 100, 16, 1,
                help="Aumente para ficar mais rígido e reduzir falsos positivos. Diminua para detectar mais bolhas."
            )

        with colp3:
            raio_minimo = st.slider(
                "Raio mínimo da bolha (px)",
                4, 80, 6, 1,
                help="Ignora bolhas muito pequenas abaixo deste valor."
            )
            raio_maximo = st.slider(
                "Raio máximo da bolha (px)",
                20, 300, 85, 1,
                help="Evita círculos muito grandes e imprecisos."
            )

        try:
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)

            # --------------------------------------------------
            # CALIBRAÇÃO
            # --------------------------------------------------
            barra_px_auto, img_calibracao = detectar_barra_escala_px(img_original_bgr)

            st.markdown("## Calibração")
            c1, c2 = st.columns(2)

            with c1:
                if barra_px_auto is not None:
                    st.success(f"Barra de 1,0 mm detectada automaticamente: {barra_px_auto} px")
                else:
                    st.warning("Barra não detectada automaticamente.")

            with c2:
                px_per_mm_manual = st.number_input(
                    "Pixels por mm (ajuste manual, se necessário)",
                    min_value=1.0,
                    value=float(barra_px_auto) if barra_px_auto is not None else 100.0,
                    step=1.0
                )

            px_per_mm = px_per_mm_manual

            st.image(
                cv_to_pil(img_calibracao),
                caption="Imagem de calibração com a barra de 1,0 mm marcada em verde",
                use_container_width=True
            )

            # --------------------------------------------------
            # FILTROS
            # --------------------------------------------------
            f_clahe = filtro_clahe(img_original_bgr)
            f_fundo = filtro_correcao_fundo(img_original_bgr)
            f_dog = filtro_dog(img_original_bgr)
            f_canny = filtro_canny(img_original_bgr)
            f_adapt = filtro_adaptativo(img_original_bgr)
            f_grad = filtro_gradiente(img_original_bgr)

            f_clahe_m = mascarar_regioes_fixas(f_clahe)
            f_fundo_m = mascarar_regioes_fixas(f_fundo)
            f_dog_m = mascarar_regioes_fixas(f_dog)
            f_canny_m = mascarar_regioes_fixas(f_canny)
            f_adapt_m = mascarar_regioes_fixas(f_adapt)
            f_grad_m = mascarar_regioes_fixas(f_grad)

            st.markdown("## Tratamento inicial")

            l1, l2, l3 = st.columns(3)
            with l1:
                st.caption("Imagem original")
                st.image(img_original_pil, use_container_width=True)
            with l2:
                st.caption("CLAHE")
                st.image(f_clahe, use_container_width=True)
            with l3:
                st.caption("Correção de fundo")
                st.image(f_fundo, use_container_width=True)

            l4, l5, l6 = st.columns(3)
            with l4:
                st.caption("DoG")
                st.image(f_dog, use_container_width=True)
            with l5:
                st.caption("Canny")
                st.image(f_canny, use_container_width=True)
            with l6:
                st.caption("Threshold adaptativo")
                st.image(f_adapt, use_container_width=True)

            st.image(f_grad, caption="Gradiente de borda", use_container_width=True)

            # --------------------------------------------------
            # DETECÇÃO HÍBRIDA
            # --------------------------------------------------
            circulos_hough_1 = detectar_bolhas_hough(
                f_clahe_m,
                sensibilidade_busca,
                distancia_minima,
                forca_borda,
                rigor_confirmacao,
                raio_minimo,
                raio_maximo
            )

            circulos_hough_2 = detectar_bolhas_hough(
                f_fundo_m,
                sensibilidade_busca,
                distancia_minima,
                forca_borda,
                rigor_confirmacao,
                raio_minimo,
                raio_maximo
            )

            circulos_hough_3 = detectar_bolhas_hough(
                f_grad_m,
                sensibilidade_busca,
                distancia_minima,
                forca_borda,
                rigor_confirmacao,
                raio_minimo,
                raio_maximo
            )

            circulos_contorno_1 = detectar_bolhas_contorno(f_adapt_m, raio_minimo, raio_maximo)
            circulos_contorno_2 = detectar_bolhas_contorno(f_canny_m, raio_minimo, raio_maximo)
            circulos_watershed = detectar_bolhas_watershed(f_fundo_m, raio_minimo, raio_maximo)

            circulos = (
                circulos_hough_1
                + circulos_hough_2
                + circulos_hough_3
                + circulos_contorno_1
                + circulos_contorno_2
                + circulos_watershed
            )

            circulos = remover_circulos_duplicados(circulos, fator=0.72)
            circulos = filtrar_circulos_borda(circulos, img_original_bgr.shape)
            circulos = remover_circulos_muito_grandes(circulos, img_original_bgr.shape)
            circulos = filtrar_por_assinatura_radial(f_fundo, circulos)
            circulos = remover_circulos_duplicados(circulos, fator=0.78)

            st.markdown("## Diagnóstico da detecção")
            d1, d2, d3, d4, d5, d6, d7 = st.columns(7)
            d1.metric("Hough CLAHE", len(circulos_hough_1))
            d2.metric("Hough Fundo", len(circulos_hough_2))
            d3.metric("Hough Grad.", len(circulos_hough_3))
            d4.metric("Contorno adapt.", len(circulos_contorno_1))
            d5.metric("Contorno Canny", len(circulos_contorno_2))
            d6.metric("Watershed", len(circulos_watershed))
            d7.metric("Final", len(circulos))

            if not circulos:
                st.warning("Nenhuma bolha foi detectada com os parâmetros atuais.")
                return

            # --------------------------------------------------
            # RESULTADOS
            # --------------------------------------------------
            df = montar_dataframe_medidas(circulos, px_per_mm)

            total_bolhas = len(df)
            media_um = df["Diâmetro (µm)"].mean()
            mediana_um = df["Diâmetro (µm)"].median()
            minimo_um = df["Diâmetro (µm)"].min()
            maximo_um = df["Diâmetro (µm)"].max()

            maiores_500 = df[df["Diâmetro (µm)"] > 500]
            qtd_maiores_500 = len(maiores_500)
            pct_maiores_500 = (qtd_maiores_500 / total_bolhas) * 100 if total_bolhas > 0 else 0

            img_anotada = desenhar_bolhas(img_original_bgr, circulos, px_per_mm)

            st.markdown("## Imagem com bolhas medidas")
            st.image(cv_to_pil(img_anotada), use_container_width=True)

            st.markdown("## Indicadores")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Quantidade de bolhas", f"{total_bolhas}")
            m2.metric("Média (µm)", f"{media_um:.1f}")
            m3.metric("Mediana (µm)", f"{mediana_um:.1f}")
            m4.metric("> 500 µm", f"{qtd_maiores_500}")

            m5, m6, m7 = st.columns(3)
            m5.metric("Mínimo (µm)", f"{minimo_um:.1f}")
            m6.metric("Máximo (µm)", f"{maximo_um:.1f}")
            m7.metric("% > 500 µm", f"{pct_maiores_500:.2f}%")

            fig_bar, fig_curve, contagens, labels = plotar_distribuicao(df)

            st.markdown("## Distribuição do tamanho das bolhas")
            st.pyplot(fig_bar)

            st.markdown("## Curva de distribuição")
            st.pyplot(fig_curve)

            resumo_faixas = pd.DataFrame({
                "Faixa (µm)": labels,
                "Quantidade": contagens
            })

            st.markdown("## Quantidade por intervalo")
            st.dataframe(resumo_faixas, use_container_width=True)

            st.markdown("## Tabela de bolhas medidas")
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
