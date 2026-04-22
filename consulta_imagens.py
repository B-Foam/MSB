import io
import requests
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt


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
# MÁSCARAS FIXAS
# ============================================================
def criar_mascara_regioes_excluidas(shape, barra_info=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # topo com texto
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
# PRÉ-PROCESSAMENTO BASE
# ============================================================
def gerar_imagens_base(img_bgr, mask_exclusao):
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_full = clahe.apply(gray_full)

    bilateral_full = cv2.bilateralFilter(
        clahe_full,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )

    gray_vis = gray_full.copy()
    clahe_vis = clahe_full.copy()
    bilateral_vis = bilateral_full.copy()

    bilateral_proc = aplicar_mascara_exclusao(bilateral_full, mask_exclusao, valor=255)

    return gray_vis, clahe_vis, bilateral_vis, bilateral_proc


# ============================================================
# MÁSCARAS CANDIDATAS
# ============================================================
def gerar_mascaras_candidatas(img_gray, mask_exclusao):
    masks = {}

    # Otsu
    _, otsu = cv2.threshold(
        img_gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    otsu[mask_exclusao == 255] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
    masks["Otsu"] = otsu

    # Adaptive 31
    ad31 = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        2
    )
    ad31[mask_exclusao == 255] = 0
    ad31 = cv2.morphologyEx(ad31, cv2.MORPH_OPEN, kernel, iterations=1)
    ad31 = cv2.morphologyEx(ad31, cv2.MORPH_CLOSE, kernel, iterations=1)
    masks["Adaptativo 31"] = ad31

    # Adaptive 51
    ad51 = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        3
    )
    ad51[mask_exclusao == 255] = 0
    ad51 = cv2.morphologyEx(ad51, cv2.MORPH_OPEN, kernel, iterations=1)
    ad51 = cv2.morphologyEx(ad51, cv2.MORPH_CLOSE, kernel, iterations=1)
    masks["Adaptativo 51"] = ad51

    return masks


# ============================================================
# ESCOLHA AUTOMÁTICA DA MELHOR MÁSCARA
# ============================================================
def avaliar_mascara(mask, shape):
    h, w = shape[:2]
    total = h * w

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    areas = []
    n_borda = 0

    for i in range(1, n_labels):
        x, y, ww, hh, area = stats[i]
        if area < 20:
            continue

        areas.append(area)

        toca_borda = (x <= 1) or (y <= 1) or (x + ww >= w - 1) or (y + hh >= h - 1)
        if toca_borda:
            n_borda += 1

    if len(areas) == 0:
        return {
            "score": -1e9,
            "n_comp": 0,
            "media_area": 0,
            "std_area": 0,
            "frac_borda": 1.0
        }

    n_comp = len(areas)
    media_area = float(np.mean(areas))
    std_area = float(np.std(areas))
    frac_borda = n_borda / max(n_comp, 1)

    # Queremos evitar zero componentes, evitar milhares de pontos minúsculos,
    # e evitar tudo grudado em poucas regiões enormes.
    score = 0.0
    score += -abs(n_comp - 180) * 0.8
    score += media_area * 0.08
    score += -std_area * 0.01
    score += -frac_borda * 120

    return {
        "score": score,
        "n_comp": n_comp,
        "media_area": media_area,
        "std_area": std_area,
        "frac_borda": frac_borda
    }


def escolher_melhor_mascara(masks, shape):
    melhor_nome = None
    melhor_mask = None
    melhor_eval = None
    melhor_score = -1e18

    avaliacoes = {}

    for nome, mask in masks.items():
        ev = avaliar_mascara(mask, shape)
        avaliacoes[nome] = ev
        if ev["score"] > melhor_score:
            melhor_score = ev["score"]
            melhor_nome = nome
            melhor_mask = mask
            melhor_eval = ev

    return melhor_nome, melhor_mask, melhor_eval, avaliacoes


# ============================================================
# WATERSHED
# ============================================================
def processar_watershed(binary_mask, img_gray, marker_factor):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, sure_fg = cv2.threshold(dist_transform, marker_factor * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)

    return markers, dist_norm, sure_fg, sure_bg


# ============================================================
# EXTRAÇÃO DAS BOLHAS A PARTIR DOS MARKERS
# ============================================================
def extrair_bolhas_dos_markers(markers, shape, px_per_mm=None):
    h, w = shape[:2]
    bolhas = []

    labels_unicos = np.unique(markers)

    for lab in labels_unicos:
        if lab <= 1:
            continue

        reg = np.uint8(markers == lab) * 255
        cnts, _ = cv2.findContours(reg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            continue

        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area < 80:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        circularidade = 4 * np.pi * area / (peri * peri)

        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull)
        if area_hull <= 0:
            continue

        solidez = area / area_hull

        x, y, ww, hh = cv2.boundingRect(cnt)

        # rejeitar regiões que tocam muito a borda
        if x <= 2 or y <= 2 or x + ww >= w - 2 or y + hh >= h - 2:
            continue

        # medidas principais
        area_eq = area
        diam_eq_px = np.sqrt((4.0 * area_eq) / np.pi)
        raio_eq_px = diam_eq_px / 2.0

        if diam_eq_px < 8:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        diam_eq_um = None
        if px_per_mm is not None and px_per_mm > 0:
            diam_eq_um = (diam_eq_px / px_per_mm) * 1000.0

        bolhas.append({
            "contorno": cnt,
            "area_px2": float(area),
            "circularidade": float(circularidade),
            "solidez": float(solidez),
            "cx": float(cx),
            "cy": float(cy),
            "diam_eq_px": float(diam_eq_px),
            "raio_eq_px": float(raio_eq_px),
            "diam_eq_um": None if diam_eq_um is None else float(diam_eq_um),
            "bbox": (x, y, ww, hh)
        })

    return bolhas


# ============================================================
# DESENHO
# ============================================================
def desenhar_markers_coloridos(markers):
    labels = markers.copy()
    labels[labels < 0] = 0

    max_label = np.max(labels)
    if max_label <= 0:
        return np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    norm = (labels.astype(np.float32) / max_label * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[markers == -1] = [255, 255, 255]
    return color


def desenhar_bolhas_segmentadas(img_bgr, bolhas):
    out = img_bgr.copy()
    rng = np.random.default_rng(42)

    for b in bolhas:
        color = tuple(int(v) for v in rng.integers(60, 256, size=3))
        cnt = b["contorno"]
        cv2.drawContours(out, [cnt], -1, color, 2)

    cv2.putText(
        out,
        f"Bolhas detectadas: {len(bolhas)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return out


# ============================================================
# RESUMO
# ============================================================
def montar_resumo_bolhas(bolhas):
    if not bolhas:
        return pd.DataFrame([{
            "Bolhas detectadas": 0,
            "Area media (px2)": 0.0,
            "Circularidade media": 0.0,
            "Solidez media": 0.0,
            "Diametro equivalente medio (px)": 0.0,
            "Diametro equivalente medio (um)": 0.0
        }])

    areas = [b["area_px2"] for b in bolhas]
    circs = [b["circularidade"] for b in bolhas]
    sols = [b["solidez"] for b in bolhas]
    dpx = [b["diam_eq_px"] for b in bolhas]
    dum = [b["diam_eq_um"] for b in bolhas if b["diam_eq_um"] is not None]

    return pd.DataFrame([{
        "Bolhas detectadas": len(bolhas),
        "Area media (px2)": round(float(np.mean(areas)), 2),
        "Circularidade media": round(float(np.mean(circs)), 4),
        "Solidez media": round(float(np.mean(sols)), 4),
        "Diametro equivalente medio (px)": round(float(np.mean(dpx)), 2),
        "Diametro equivalente medio (um)": round(float(np.mean(dum)), 2) if len(dum) > 0 else 0.0
    }])


# ============================================================
# GRÁFICOS
# ============================================================
def plotar_distribuicao(bolhas):
    diametros = [b["diam_eq_um"] for b in bolhas if b["diam_eq_um"] is not None]
    if len(diametros) == 0:
        return None, None, None

    diametros = np.array(diametros)
    max_d = max(600, int(np.ceil(diametros.max() / 100.0) * 100))
    bins = list(range(0, max_d + 100, 100))
    contagens, bordas = np.histogram(diametros, bins=bins)
    labels = [f"{int(bordas[i])}-{int(bordas[i+1])}" for i in range(len(bordas) - 1)]
    centros = [(bordas[i] + bordas[i + 1]) / 2 for i in range(len(bordas) - 1)]

    fig_bar, ax_bar = plt.subplots(figsize=(8, 3.5))
    ax_bar.bar(labels, contagens)
    ax_bar.set_title("Quantidade de bolhas por faixa")
    ax_bar.set_xlabel("Faixa de diâmetro (µm)")
    ax_bar.set_ylabel("Quantidade")
    plt.xticks(rotation=45)
    plt.tight_layout()

    fig_curve, ax_curve = plt.subplots(figsize=(8, 3.5))
    ax_curve.plot(centros, contagens, marker="o")
    ax_curve.set_title("Curva de distribuição")
    ax_curve.set_xlabel("Diâmetro (µm)")
    ax_curve.set_ylabel("Quantidade")
    plt.tight_layout()

    tabela = pd.DataFrame({
        "Faixa (µm)": labels,
        "Quantidade": contagens
    })

    return fig_bar, fig_curve, tabela


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

        # ÚNICO AJUSTE PERMITIDO AO USUÁRIO
        marker_factor = st.selectbox(
            "Sensibilidade interna do Watershed",
            [0.20, 0.28, 0.35],
            index=1,
            help="Valores menores capturam mais bolhas pequenas; valores maiores ficam mais conservadores."
        )

        try:
            url_imagem = montar_url_publica(escolhido)
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)

            barra_px_auto, img_calibracao, barra_info = detectar_barra_escala_px(img_original_bgr)
            mask_exclusao = criar_mascara_regioes_excluidas(img_original_bgr.shape, barra_info)

            gray_vis, clahe_vis, bilateral_vis, bilateral_proc = gerar_imagens_base(
                img_original_bgr,
                mask_exclusao
            )

            masks = gerar_mascaras_candidatas(bilateral_proc, mask_exclusao)
            melhor_nome, melhor_mask, melhor_eval, avaliacoes = escolher_melhor_mascara(
                masks,
                img_original_bgr.shape
            )

            markers, dist_norm, sure_fg, sure_bg = processar_watershed(
                melhor_mask,
                bilateral_proc,
                marker_factor
            )

            bolhas = extrair_bolhas_dos_markers(
                markers,
                img_original_bgr.shape,
                px_per_mm=barra_px_auto
            )

            img_markers = desenhar_markers_coloridos(markers)
            img_bolhas = desenhar_bolhas_segmentadas(img_original_bgr, bolhas)

            st.markdown("## Calibração")
            if barra_px_auto is not None:
                st.success(f"Barra de 1,0 mm detectada automaticamente: {barra_px_auto} px")
            else:
                st.warning("Barra não detectada automaticamente.")

            st.image(cv_to_pil(img_calibracao), caption="Barra de 1,0 mm detectada", width=460)

            st.markdown("## Pré-processamento automático")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("Imagem original")
                st.image(img_original_pil, width=240)
            with c2:
                st.caption("CLAHE")
                st.image(clahe_vis, width=240)
            with c3:
                st.caption("Filtro bilateral")
                st.image(bilateral_vis, width=240)

            st.markdown("## Máscaras candidatas")
            m1, m2, m3 = st.columns(3)
            nomes = list(masks.keys())
            with m1:
                st.caption(nomes[0])
                st.image(masks[nomes[0]], width=240)
            with m2:
                st.caption(nomes[1])
                st.image(masks[nomes[1]], width=240)
            with m3:
                st.caption(nomes[2])
                st.image(masks[nomes[2]], width=240)

            st.info(f"Máscara escolhida automaticamente: **{melhor_nome}**")

            st.markdown("## Watershed")
            w1, w2, w3 = st.columns(3)
            with w1:
                st.caption("Mapa de distância")
                st.image(dist_norm, width=240)
            with w2:
                st.caption("Foreground seguro")
                st.image(sure_fg, width=240)
            with w3:
                st.caption("Markers / regiões")
                st.image(cv_to_pil(img_markers), width=240)

            st.markdown("## Resultado automático")
            st.image(cv_to_pil(img_bolhas), width=760)

            st.markdown("## Diagnóstico")
            d1, d2, d3 = st.columns(3)
            d1.metric("Componentes da máscara", melhor_eval["n_comp"])
            d2.metric("Marker factor", marker_factor)
            d3.metric("Bolhas detectadas", len(bolhas))

            st.markdown("## Resumo")
            st.dataframe(montar_resumo_bolhas(bolhas), use_container_width=True)

            if len(bolhas) > 0:
                maiores_500 = [b for b in bolhas if b["diam_eq_um"] is not None and b["diam_eq_um"] > 500]
                pct_500 = 100.0 * len(maiores_500) / len(bolhas)

                k1, k2, k3 = st.columns(3)
                k1.metric("Bolhas > 500 µm", len(maiores_500))
                k2.metric("% > 500 µm", f"{pct_500:.2f}%")
                k3.metric("Bolhas totais", len(bolhas))

                fig_bar, fig_curve, tabela = plotar_distribuicao(bolhas)
                if fig_bar is not None:
                    st.markdown("## Distribuição granulométrica")
                    st.pyplot(fig_bar)
                    st.markdown("## Curva")
                    st.pyplot(fig_curve)
                    st.markdown("## Quantidade por faixa")
                    st.dataframe(tabela, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
