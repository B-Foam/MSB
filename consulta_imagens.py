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

    # faixa superior com texto
    mask[0:int(h * 0.05), :] = 255

    # barra de escala e canto inferior esquerdo
    mask[int(h * 0.78):h, 0:int(w * 0.22)] = 255

    return mask


def aplicar_mascara_exclusao(img_gray, mask_exclusao):
    out = img_gray.copy()
    out[mask_exclusao == 255] = 255
    return out


# ============================================================
# PRÉ-PROCESSAMENTO OTIMIZADO
# ============================================================
def preprocessar_imagem_otimizado(img_bgr, mask_exclusao):
    """
    Pipeline simplificado para gerar uma máscara mais limpa:
    1) escala de cinza
    2) máscara das regiões fixas
    3) CLAHE
    4) filtro bilateral
    5) Otsu
    6) fechamento morfológico leve
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_mask = aplicar_mascara_exclusao(gray, mask_exclusao)

    # 1) CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_mask)

    # 2) Bilateral
    smooth = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # 3) Otsu
    _, th = cv2.threshold(
        smooth,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # remove regiões excluídas
    th[mask_exclusao == 255] = 0

    # 4) fechamento morfológico leve
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    return gray, enhanced, smooth, th, cleaned


# ============================================================
# WATERSHED
# ============================================================
def segmentar_por_watershed(bin_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    sure_bg = cv2.dilate(bin_img, kernel, iterations=2)

    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, sure_fg = cv2.threshold(dist, 0.28 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_color = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    return markers, dist_norm, sure_fg, sure_bg


# ============================================================
# EXTRAÇÃO E FILTRO DOS COMPONENTES
# ============================================================
def extrair_bolhas_dos_markers(markers, raio_min_px, raio_max_px):
    bolhas = []

    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue

        mask = np.uint8(markers == marker_id) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue

            hull = cv2.convexHull(cnt)
            area_hull = cv2.contourArea(hull)

            if area_hull <= 0:
                continue

            circularidade = 4 * np.pi * area / (peri * peri)
            solidez = area / area_hull

            (x, y), r = cv2.minEnclosingCircle(cnt)

            if r < raio_min_px or r > raio_max_px:
                continue

            bolhas.append({
                "x": int(x),
                "y": int(y),
                "r": float(r),
                "area_px": float(area),
                "circularidade": float(circularidade),
                "solidez": float(solidez),
                "contorno": cnt
            })

    return bolhas


def filtrar_bolhas_segmentadas(
    bolhas,
    circularidade_min,
    solidez_min,
    area_min_px,
    area_max_px
):
    finais = []

    for b in bolhas:
        if b["area_px"] < area_min_px:
            continue
        if b["area_px"] > area_max_px:
            continue
        if b["circularidade"] < circularidade_min:
            continue
        if b["solidez"] < solidez_min:
            continue
        finais.append(b)

    return finais


def remover_bolhas_muito_proximas(bolhas, fator=0.55):
    if not bolhas:
        return []

    bolhas = sorted(bolhas, key=lambda b: b["r"], reverse=True)
    finais = []

    for b in bolhas:
        manter = True

        for f in finais:
            dist = ((b["x"] - f["x"]) ** 2 + (b["y"] - f["y"]) ** 2) ** 0.5
            if dist < fator * min(b["r"], f["r"]):
                manter = False
                break

        if manter:
            finais.append(b)

    return finais


def filtrar_borda_bolhas(bolhas, shape):
    h, w = shape[:2]
    finais = []

    for b in bolhas:
        x, y, r = b["x"], b["y"], b["r"]

        if x - r <= 4 or y - r <= 4 or x + r >= w - 4 or y + r >= h - 4:
            continue

        finais.append(b)

    return finais


# ============================================================
# VISUALIZAÇÕES
# ============================================================
def desenhar_bolhas(img_bgr, bolhas, px_per_mm):
    out = img_bgr.copy()

    for b in bolhas:
        x, y, r = int(b["x"]), int(b["y"]), int(b["r"])
        diam_um = (2 * r / px_per_mm) * 1000.0

        cor = (0, 255, 0)
        if diam_um > 500:
            cor = (255, 0, 0)

        cv2.circle(out, (x, y), r, cor, 2)

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
        f"Bolhas detectadas: {len(bolhas)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return out


def desenhar_segmentacao(markers):
    labels = markers.copy()
    labels[labels < 0] = 0

    max_label = np.max(labels)
    if max_label <= 0:
        return np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    norm = (labels.astype(np.float32) / max_label * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[markers == -1] = [255, 255, 255]
    return color


# ============================================================
# DATAFRAME E GRÁFICOS
# ============================================================
def montar_dataframe_medidas(bolhas, px_per_mm):
    dados = []

    for i, b in enumerate(bolhas, start=1):
        raio_px = b["r"]
        diam_um = (2 * raio_px / px_per_mm) * 1000.0
        raio_um = (raio_px / px_per_mm) * 1000.0
        area_um2 = (b["area_px"] / (px_per_mm ** 2)) * 1_000_000

        dados.append({
            "Bolha": i,
            "Centro X (px)": b["x"],
            "Centro Y (px)": b["y"],
            "Raio (px)": round(raio_px, 2),
            "Raio (µm)": round(raio_um, 2),
            "Diâmetro (µm)": round(diam_um, 2),
            "Área (px²)": round(b["area_px"], 2),
            "Área (µm²)": round(area_um2, 2),
            "Circularidade": round(b["circularidade"], 4),
            "Solidez": round(b["solidez"], 4)
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

        st.markdown("## Configurações da segmentação")

        col1, col2, col3 = st.columns(3)

        with col1:
            raio_minimo = st.slider("Raio mínimo da bolha (px)", 4, 80, 6, 1)
            raio_maximo = st.slider("Raio máximo da bolha (px)", 20, 300, 90, 1)

        with col2:
            circularidade_min = st.slider("Circularidade mínima", 0.10, 1.00, 0.35, 0.01)
            solidez_min = st.slider("Solidez mínima", 0.10, 1.00, 0.70, 0.01)

        with col3:
            area_min_px = st.slider("Área mínima da bolha (px²)", 10, 5000, 80, 10)
            area_max_px = st.slider("Área máxima da bolha (px²)", 100, 100000, 12000, 100)

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

            # Pré-processamento
            mask_exclusao = criar_mascara_regioes_excluidas(img_original_bgr.shape)

            gray, enhanced, smooth, th, cleaned = preprocessar_imagem_otimizado(
                img_original_bgr,
                mask_exclusao
            )

            st.markdown("## Tratamento inicial")

            p1, p2, p3 = st.columns(3)
            with p1:
                st.caption("Imagem original")
                st.image(img_original_pil, use_container_width=True)
            with p2:
                st.caption("CLAHE")
                st.image(enhanced, use_container_width=True)
            with p3:
                st.caption("Filtro bilateral")
                st.image(smooth, use_container_width=True)

            p4, p5 = st.columns(2)
            with p4:
                st.caption("Threshold de Otsu")
                st.image(th, use_container_width=True)
            with p5:
                st.caption("Máscara final para segmentação")
                st.image(cleaned, use_container_width=True)

            # Watershed
            markers, dist_norm, sure_fg, sure_bg = segmentar_por_watershed(cleaned)
            img_segmentacao = desenhar_segmentacao(markers)

            st.markdown("## Segmentação")
            s1, s2, s3 = st.columns(3)
            with s1:
                st.caption("Mapa de distância")
                st.image(dist_norm, use_container_width=True)
            with s2:
                st.caption("Foreground seguro")
                st.image(sure_fg, use_container_width=True)
            with s3:
                st.caption("Watershed")
                st.image(cv_to_pil(img_segmentacao), use_container_width=True)

            # Extração / filtro
            bolhas_brutas = extrair_bolhas_dos_markers(markers, raio_minimo, raio_maximo)
            bolhas_filtradas = filtrar_bolhas_segmentadas(
                bolhas_brutas,
                circularidade_min,
                solidez_min,
                area_min_px,
                area_max_px
            )
            bolhas_filtradas = filtrar_borda_bolhas(bolhas_filtradas, img_original_bgr.shape)
            bolhas_filtradas = remover_bolhas_muito_proximas(bolhas_filtradas, fator=0.55)

            st.markdown("## Diagnóstico da segmentação")
            d1, d2, d3 = st.columns(3)
            d1.metric("Componentes brutos", len(bolhas_brutas))
            d2.metric("Após filtros geométricos", len(bolhas_filtradas))
            d3.metric("Final", len(bolhas_filtradas))

            if not bolhas_filtradas:
                st.warning("Nenhuma bolha foi detectada com os parâmetros atuais.")
                return

            # Resultados
            df = montar_dataframe_medidas(bolhas_filtradas, px_per_mm)

            total_bolhas = len(df)
            media_um = df["Diâmetro (µm)"].mean()
            mediana_um = df["Diâmetro (µm)"].median()
            minimo_um = df["Diâmetro (µm)"].min()
            maximo_um = df["Diâmetro (µm)"].max()

            maiores_500 = df[df["Diâmetro (µm)"] > 500]
            qtd_maiores_500 = len(maiores_500)
            pct_maiores_500 = (qtd_maiores_500 / total_bolhas) * 100 if total_bolhas > 0 else 0

            img_anotada = desenhar_bolhas(img_original_bgr, bolhas_filtradas, px_per_mm)

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
