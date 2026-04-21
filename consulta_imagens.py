import io
import requests
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def baixar_imagem(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def detectar_barra_escala_px(img_bgr):
    """
    Detecta a barra horizontal preta no canto inferior esquerdo.
    Retorna o comprimento em pixels da barra horizontal.
    """
    h, w = img_bgr.shape[:2]

    crop = img_bgr[int(h * 0.78):h, 0:int(w * 0.35)].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Destacar regiões escuras
    _, th = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Limpeza
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor_largura = None

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / max(hh, 1)

        # Procurar segmentos horizontais longos e finos
        if ww > 60 and hh < 20 and aspect > 5:
            if melhor_largura is None or ww > melhor_largura:
                melhor_largura = ww

    return melhor_largura


def aplicar_filtro_1(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    f1 = clahe.apply(gray)
    return f1


def aplicar_filtro_2(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    f2 = clahe.apply(blur)
    return f2


def mascarar_regioes_fixas(img_gray):
    """
    Mascara a faixa superior com informações da câmera
    e a região inferior esquerda da barra de escala
    para evitar falsas detecções.
    """
    out = img_gray.copy()
    h, w = out.shape[:2]

    # Topo
    out[0:int(h * 0.05), :] = 255

    # Região da barra
    out[int(h * 0.78):h, 0:int(w * 0.22)] = 255

    return out


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


def remover_circulos_duplicados(circulos, fator=0.6):
    """
    Remove círculos muito próximos entre si.
    """
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


def desenhar_bolhas(img_bgr, circulos, px_per_mm):
    out = img_bgr.copy()

    for i, (x, y, r) in enumerate(circulos, start=1):
        diam_um = (2 * r / px_per_mm) * 1000.0
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)

        # Para não poluir demais, só escreve em bolhas maiores
        if diam_um >= 300:
            texto = f"{int(diam_um)}"
            cv2.putText(
                out,
                texto,
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

    df = pd.DataFrame(dados)
    return df


def plotar_distribuicao(df):
    diametros = df["Diâmetro (µm)"].values

    max_d = max(600, int(np.ceil(diametros.max() / 100.0) * 100))
    bins = list(range(0, max_d + 100, 100))

    contagens, bordas = np.histogram(diametros, bins=bins)
    centros = [(bordas[i] + bordas[i + 1]) / 2 for i in range(len(bordas) - 1)]

    labels = []
    for i in range(len(bordas) - 1):
        labels.append(f"{int(bordas[i])}-{int(bordas[i+1])}")

    # Gráfico de barras
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    ax_bar.bar(labels, contagens)
    ax_bar.set_title("Quantidade de bolhas por faixa de diâmetro")
    ax_bar.set_xlabel("Faixa de diâmetro (µm)")
    ax_bar.set_ylabel("Quantidade")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Curva de distribuição
    fig_curve, ax_curve = plt.subplots(figsize=(10, 4))
    ax_curve.plot(centros, contagens, marker="o")
    ax_curve.set_title("Curva de distribuição do tamanho das bolhas")
    ax_curve.set_xlabel("Diâmetro (µm)")
    ax_curve.set_ylabel("Quantidade")
    plt.tight_layout()

    return fig_bar, fig_curve, contagens, labels


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

        st.markdown("### Parâmetros de detecção")

        colp1, colp2, colp3 = st.columns(3)
        with colp1:
            dp = st.slider("dp", 1.0, 2.5, 1.2, 0.1)
            min_dist = st.slider("Distância mínima", 10, 100, 28, 1)
        with colp2:
            param1 = st.slider("param1", 20, 200, 80, 1)
            param2 = st.slider("param2", 10, 100, 22, 1)
        with colp3:
            min_radius = st.slider("Raio mínimo (px)", 5, 80, 10, 1)
            max_radius = st.slider("Raio máximo (px)", 20, 300, 95, 1)

        try:
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)

            # Calibração automática pela barra de 1 mm
            barra_px = detectar_barra_escala_px(img_original_bgr)

            st.markdown("### Calibração")
            colc1, colc2 = st.columns(2)

            with colc1:
                if barra_px is not None:
                    st.success(f"Barra de 1,0 mm detectada: {barra_px} px")
                else:
                    st.warning("Barra não detectada automaticamente.")

            with colc2:
                px_per_mm_manual = st.number_input(
                    "px por mm (ajuste manual, se necessário)",
                    min_value=1.0,
                    value=float(barra_px) if barra_px is not None else 100.0,
                    step=1.0
                )

            px_per_mm = px_per_mm_manual

            # Filtros
            filtro_1 = aplicar_filtro_1(img_original_bgr)
            filtro_2 = aplicar_filtro_2(img_original_bgr)

            filtro_2_mask = mascarar_regioes_fixas(filtro_2)

            # Detecção
            circulos = detectar_bolhas_hough(
                filtro_2_mask,
                dp=dp,
                min_dist=min_dist,
                param1=param1,
                param2=param2,
                min_radius=min_radius,
                max_radius=max_radius
            )

            circulos = remover_circulos_duplicados(circulos)

            st.markdown("### Tratamento inicial")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("Imagem original")
                st.image(img_original_pil, use_container_width=True)

            with col2:
                st.caption("Filtro 1 — contraste local")
                st.image(filtro_1, use_container_width=True)

            with col3:
                st.caption("Filtro 2 — suavização + contraste")
                st.image(filtro_2, use_container_width=True)

            if not circulos:
                st.warning("Nenhuma bolha foi detectada com os parâmetros atuais.")
                return

            # DataFrame de medidas
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

            st.markdown("### Imagem com bolhas medidas")
            st.image(cv_to_pil(img_anotada), use_container_width=True)

            st.markdown("### Indicadores")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Quantidade de bolhas", f"{total_bolhas}")
            m2.metric("Média (µm)", f"{media_um:.1f}")
            m3.metric("Mediana (µm)", f"{mediana_um:.1f}")
            m4.metric("> 500 µm", f"{qtd_maiores_500}")

            m5, m6, m7 = st.columns(3)
            m5.metric("Mínimo (µm)", f"{minimo_um:.1f}")
            m6.metric("Máximo (µm)", f"{maximo_um:.1f}")
            m7.metric("% > 500 µm", f"{pct_maiores_500:.2f}%")

            # Gráficos
            fig_bar, fig_curve, contagens, labels = plotar_distribuicao(df)

            st.markdown("### Distribuição do tamanho das bolhas")
            st.pyplot(fig_bar)

            st.markdown("### Curva de distribuição")
            st.pyplot(fig_curve)

            # Resumo por intervalo
            resumo_faixas = pd.DataFrame({
                "Faixa (µm)": labels,
                "Quantidade": contagens
            })

            st.markdown("### Quantidade por intervalo")
            st.dataframe(resumo_faixas, use_container_width=True)

            st.markdown("### Tabela de bolhas medidas")
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
