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
    gww = ww
    ghh = hh

    cv2.rectangle(img_annot, (gx, gy), (gx + gww, gy + ghh), (0, 255, 0), 2)
    cv2.line(
        img_annot,
        (gx, gy + ghh // 2),
        (gx + gww, gy + ghh // 2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        img_annot,
        f"1.0 mm = {gww} px",
        (gx, max(20, gy - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return gww, img_annot


# ============================================================
# MÁSCARAS FIXAS
# ============================================================
def mascarar_regioes_fixas(img_gray):
    out = img_gray.copy()
    h, w = out.shape[:2]

    # faixa superior com texto da câmera
    out[0:int(h * 0.05), :] = 255

    # região da barra de escala
    out[int(h * 0.78):h, 0:int(w * 0.22)] = 255

    return out


# ============================================================
# FILTROS DE PRÉ-PROCESSAMENTO
# ============================================================
def filtro_clahe(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def filtro_correcao_fundo(img_bgr):
    """
    Remove variações lentas de iluminação usando blur grande.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 25)
    corr = cv2.subtract(background, gray)
    corr = cv2.normalize(corr, None, 0, 255, cv2.NORM_MINMAX)
    return corr


def filtro_dog(img_bgr):
    """
    Difference of Gaussians para realçar anéis e bordas.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.2)
    g2 = cv2.GaussianBlur(gray, (0, 0), 3.0)
    dog = cv2.subtract(g1, g2)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    return dog


def filtro_canny(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    return edges


def filtro_adaptativo(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        3
    )
    return th


# ============================================================
# DETECÇÃO POR HOUGH
# ============================================================
def detectar_bolhas_hough(
    img_gray,
    dp,
    min_dist,
    param1,
    param2,
    min_radius,
    max_radius
):
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int)
    return circles.tolist()


# ============================================================
# DETECÇÃO POR CONTORNO / ANEL
# ============================================================
def detectar_bolhas_contorno(bin_img, min_radius, max_radius):
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

        if r < min_radius or r > max_radius:
            continue

        if circularidade >= 0.35:
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

            # remove círculos muito sobrepostos
            if dist < fator * min(r, rf):
                manter = False
                break

            # remove círculos concêntricos quase iguais
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


def filtrar_por_contraste_local(img_gray, circulos):
    """
    Mantém só círculos cujo anel tem contraste suficiente.
    Isso ajuda a reduzir falsos positivos em áreas lisas.
    """
    finais = []
    h, w = img_gray.shape[:2]

    for x, y, r in circulos:
        if r < 3:
            continue

        mask_inner = np.zeros((h, w), dtype=np.uint8)
        mask_ring = np.zeros((h, w), dtype=np.uint8)

        cv2.circle(mask_inner, (x, y), max(1, int(r * 0.55)), 255, -1)
        cv2.circle(mask_ring, (x, y), int(r), 255, 2)

        inner_vals = img_gray[mask_inner == 255]
        ring_vals = img_gray[mask_ring == 255]

        if len(inner_vals) == 0 or len(ring_vals) == 0:
            continue

        contraste = abs(float(np.mean(ring_vals)) - float(np.mean(inner_vals)))

        if contraste >= 8:
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
    labels = [f"{int(bordas[i])}-{int(bordas[i+1])}" for i in range(len(bordas)-1)]

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

        st.markdown("## Parâmetros de detecção")

        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            dp = st.slider("dp", 1.0, 2.5, 1.2, 0.1)
            min_dist = st.slider("Distância mínima", 8, 120, 20, 1)
        with colp2:
            param1 = st.slider("param1", 20, 200, 90, 1)
            param2 = st.slider("param2", 8, 100, 18, 1)
        with colp3:
            min_radius = st.slider("Raio mínimo (px)", 4, 80, 7, 1)
            max_radius = st.slider("Raio máximo (px)", 20, 300, 120, 1)

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
                    "px por mm (ajuste manual, se necessário)",
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

            f_clahe_m = mascarar_regioes_fixas(f_clahe)
            f_fundo_m = mascarar_regioes_fixas(f_fundo)
            f_dog_m = mascarar_regioes_fixas(f_dog)
            f_canny_m = mascarar_regioes_fixas(f_canny)
            f_adapt_m = mascarar_regioes_fixas(f_adapt)

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

            # --------------------------------------------------
            # DETECÇÃO HÍBRIDA
            # --------------------------------------------------
            circulos_hough_1 = detectar_bolhas_hough(
                f_clahe_m,
                dp=dp,
                min_dist=min_dist,
                param1=param1,
                param2=param2,
                min_radius=min_radius,
                max_radius=max_radius
            )

            circulos_hough_2 = detectar_bolhas_hough(
                f_fundo_m,
                dp=dp,
                min_dist=min_dist,
                param1=param1,
                param2=param2,
                min_radius=min_radius,
                max_radius=max_radius
            )

            circulos_hough_3 = detectar_bolhas_hough(
                f_dog_m,
                dp=dp,
                min_dist=min_dist,
                param1=param1,
                param2=param2,
                min_radius=min_radius,
                max_radius=max_radius
            )

            circulos_contorno_1 = detectar_bolhas_contorno(f_adapt_m, min_radius, max_radius)
            circulos_contorno_2 = detectar_bolhas_contorno(f_canny_m, min_radius, max_radius)

            circulos = (
                circulos_hough_1
                + circulos_hough_2
                + circulos_hough_3
                + circulos_contorno_1
                + circulos_contorno_2
            )

            circulos = remover_circulos_duplicados(circulos, fator=0.72)
            circulos = filtrar_circulos_borda(circulos, img_original_bgr.shape)
            circulos = filtrar_por_contraste_local(f_clahe, circulos)
            circulos = remover_circulos_duplicados(circulos, fator=0.78)

            st.markdown("## Diagnóstico da detecção")
            d1, d2, d3, d4, d5, d6 = st.columns(6)
            d1.metric("Hough CLAHE", len(circulos_hough_1))
            d2.metric("Hough Fundo", len(circulos_hough_2))
            d3.metric("Hough DoG", len(circulos_hough_3))
            d4.metric("Contorno adapt.", len(circulos_contorno_1))
            d5.metric("Contorno Canny", len(circulos_contorno_2))
            d6.metric("Final", len(circulos))

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
