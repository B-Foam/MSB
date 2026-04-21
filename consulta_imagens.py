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

    # Busca regiões escuras
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

        # barra horizontal longa e fina
        if ww > 60 and hh < 20 and aspect > 5:
            if area > melhor_area:
                melhor_area = area
                melhor = (x, y, ww, hh)

    if melhor is None:
        return None, img_annot

    x, y, ww, hh = melhor

    # coordenadas globais
    gx = x0 + x
    gy = y0 + y
    gww = ww
    ghh = hh

    # desenha retângulo verde
    cv2.rectangle(img_annot, (gx, gy), (gx + gww, gy + ghh), (0, 255, 0), 2)

    # desenha linha central verde
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
# FILTROS
# ============================================================
def aplicar_filtro_1(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    f1 = clahe.apply(gray)
    return f1


def aplicar_filtro_2(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    f2 = clahe.apply(blur)
    return f2


def aplicar_filtro_3_contornos(img_bgr):
    """
    Filtro extra para destacar bordas.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(blur, cv2.CV_8U, ksize=3)
    eq = cv2.equalizeHist(lap)
    return eq


def mascarar_regioes_fixas(img_gray):
    """
    Remove da análise:
    - topo com informações da câmera
    - região inferior esquerda da barra de escala
    """
    out = img_gray.copy()
    h, w = out.shape[:2]

    out[0:int(h * 0.05), :] = 255
    out[int(h * 0.78):h, 0:int(w * 0.22)] = 255

    return out


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
# DETECÇÃO POR CONTORNO CIRCULAR
# ============================================================
def detectar_bolhas_contorno(img_gray, min_radius, max_radius):
    """
    Detecta bolhas por contornos quase circulares.
    """
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        3
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circulos = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circularidade = 4 * np.pi * area / (peri * peri)

        (x, y), r = cv2.minEnclosingCircle(cnt)

        if r < min_radius or r > max_radius:
            continue

        # tolerância razoável porque as bolhas reais não são perfeitas
        if circularidade >= 0.45:
            circulos.append([int(x), int(y), int(r)])

    return circulos


# ============================================================
# MERGE / LIMPEZA
# ============================================================
def remover_circulos_duplicados(circulos, fator=0.65):
    if not circulos:
        return []

    circulos = sorted(circulos, key=lambda c: c[2], reverse=True)
    finais = []

    for c in circulos:
        x, y, r = c
        manter = True

        for f in finais:
            xf, yf, rf = f
            dist = ((x - xf) ** 2 + (y - yf) ** 2) ** 0.5

            if dist < fator * min(r, rf):
                manter = False
                break

        if manter:
            finais.append(c)

    return finais


def filtrar_circulos_borda(circulos, shape):
    """
    Remove detecções grudadas nas bordas da imagem.
    """
    h, w = shape[:2]
    finais = []

    for x, y, r in circulos:
        if x - r <= 2 or y - r <= 2 or x + r >= w - 2 or y + r >= h - 2:
            continue
        finais.append([x, y, r])

    return finais


# ============================================================
# ANOTAÇÃO / TABELA
# ============================================================
def desenhar_bolhas(img_bgr, circulos, px_per_mm):
    out = img_bgr.copy()

    for i, (x, y, r) in enumerate(circulos, start=1):
        diam_um = (2 * r / px_per_mm) * 1000.0

        cv2.circle(out, (x, y), r, (0, 255, 0), 2)

        if diam_um >= 300:
            cv2.putText(
                out,
                f"{int(diam_um)}",
                (x - 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
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
# TELA PRINCIPAL DA CONSULTA
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
            min_dist = st.slider("Distância mínima", 8, 120, 22, 1)
        with colp2:
            param1 = st.slider("param1", 20, 200, 90, 1)
            param2 = st.slider("param2", 8, 100, 18, 1)
        with colp3:
            min_radius = st.slider("Raio mínimo (px)", 4, 80, 8, 1)
            max_radius = st.slider("Raio máximo (px)", 20, 300, 120, 1)

        try:
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)

            # Calibração
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

            # Filtros
            filtro_1 = aplicar_filtro_1(img_original_bgr)
            filtro_2 = aplicar_filtro_2(img_original_bgr)
            filtro_3 = aplicar_filtro_3_contornos(img_original_bgr)

            filtro_hough = mascarar_regioes_fixas(filtro_2)
            filtro_contorno = mascarar_regioes_fixas(filtro_3)

            st.markdown("## Tratamento inicial")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.caption("Imagem original")
                st.image(img_original_pil, use_container_width=True)

            with col2:
                st.caption("Filtro 1 — contraste local")
                st.image(filtro_1, use_container_width=True)

            with col3:
                st.caption("Filtro 2 — suavização + contraste")
                st.image(filtro_2, use_container_width=True)

            with col4:
                st.caption("Filtro 3 — realce de contornos")
                st.image(filtro_3, use_container_width=True)

            # Detecção Hough
            circulos_hough = detectar_bolhas_hough(
                filtro_hough,
                dp=dp,
                min_dist=min_dist,
                param1=param1,
                param2=param2,
                min_radius=min_radius,
                max_radius=max_radius
            )

            # Detecção por contorno
            circulos_contorno = detectar_bolhas_contorno(
                filtro_contorno,
                min_radius=min_radius,
                max_radius=max_radius
            )

            # Junta ambos
            circulos = circulos_hough + circulos_contorno
            circulos = remover_circulos_duplicados(circulos, fator=0.65)
            circulos = filtrar_circulos_borda(circulos, img_original_bgr.shape)

            st.markdown("## Diagnóstico da detecção")
            d1, d2, d3 = st.columns(3)
            d1.metric("Detectadas por Hough", len(circulos_hough))
            d2.metric("Detectadas por contorno", len(circulos_contorno))
            d3.metric("Após união e limpeza", len(circulos))

            if not circulos:
                st.warning("Nenhuma bolha foi detectada com os parâmetros atuais.")
                return

            # Medidas
            df = montar_dataframe_medidas(circulos, px_per_mm)

            total_bolhas = len(df)
            media_um = df["Diâmetro (µm)"].mean()
            mediana_um = df["Diâmetro (µm)"].median()
            minimo_um = df["Diâmetro (µm)"].min()
            maximo_um = df["Diâmetro (µm)"].max()

            maiores_500 = df[df["Diâmetro (µm)"] > 500]
            qtd_maiores_500 = len(maiores_500)
            pct_maiores_500 = (qtd_maiores_500 / total_bolhas) * 100 if total_bolhas > 0 else 0

            # Imagem anotada
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
