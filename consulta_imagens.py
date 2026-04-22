import io
import math
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
def baixar_imagem(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    if len(img_cv.shape) == 2:
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


# ============================================================
# CALIBRAÇÃO DA BARRA DE 1,0 mm
# ============================================================
def detectar_barra_escala_px(img_bgr: np.ndarray):
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

        if ww > 60 and hh < 20 and aspect > 5 and area > melhor_area:
            melhor_area = area
            melhor = (x, y, ww, hh)

    if melhor is None:
        return None, img_annot, None

    x, y, ww, hh = melhor
    gx = x0 + x
    gy = y0 + y

    cv2.rectangle(img_annot, (gx, gy), (gx + ww, gy + hh), (0, 255, 0), 2)
    cv2.line(img_annot, (gx, gy + hh // 2), (gx + ww, gy + hh // 2), (0, 255, 0), 2)
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
# ROI CIRCULAR E MÁSCARAS
# ============================================================
def criar_mascara_circular(shape, raio_frac=0.43):
    h, w = shape[:2]
    cx = w // 2
    cy = h // 2
    r = int(min(h, w) * raio_frac)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask, {"cx": cx, "cy": cy, "r": r}


def criar_mascara_regioes_excluidas(shape, barra_info=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # texto superior
    mask[0:int(h * 0.035), :] = 255

    # barra de escala
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


def combinar_mascaras_roi(mask_circular, mask_excluir):
    roi = np.zeros_like(mask_circular)
    roi[(mask_circular == 255) & (mask_excluir == 0)] = 255
    return roi


def aplicar_roi(img_gray, mask_roi, valor_fora=0):
    out = img_gray.copy()
    out[mask_roi == 0] = valor_fora
    return out


def desenhar_roi_circular(img_bgr, roi_info):
    out = img_bgr.copy()
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 0), 2)
    return out


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================
def preprocessar_base(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    bilateral = cv2.bilateralFilter(clahe_img, 9, 75, 75)

    blur_ref = cv2.GaussianBlur(bilateral, (0, 0), 1.2)
    sharpen = cv2.addWeighted(bilateral, 1.35, blur_ref, -0.35, 0)

    gray_roi = aplicar_roi(gray, mask_roi, valor_fora=0)
    clahe_roi = aplicar_roi(clahe_img, mask_roi, valor_fora=0)
    bilateral_roi = aplicar_roi(bilateral, mask_roi, valor_fora=0)
    sharpen_roi = aplicar_roi(sharpen, mask_roi, valor_fora=0)

    return {
        "gray_vis": gray,
        "clahe_vis": clahe_img,
        "bilateral_vis": bilateral,
        "sharpen_vis": sharpen,
        "gray_roi": gray_roi,
        "clahe_roi": clahe_roi,
        "bilateral_roi": bilateral_roi,
        "sharpen_roi": sharpen_roi
    }


# ============================================================
# WATERSHED
# ============================================================
def detectar_bolhas_watershed(img_gray, mask_roi, marker_factor=0.28):
    img_roi = cv2.bitwise_and(img_gray, img_gray, mask=mask_roi)

    thresh = cv2.adaptiveThreshold(
        img_roi,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    thresh[mask_roi == 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, sure_fg = cv2.threshold(dist, marker_factor * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers[mask_roi == 0] = 1

    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)

    return {
        "thresh": thresh,
        "opening": opening,
        "dist_norm": dist_norm,
        "sure_fg": sure_fg,
        "sure_bg": sure_bg,
        "markers": markers
    }


# ============================================================
# EXTRAÇÃO DAS BOLHAS
# ============================================================
def extrair_bolhas_dos_markers(markers, shape, roi_info, px_per_mm=None):
    h, w = shape[:2]
    bolhas = []

    labels_unicos = np.unique(markers)

    for label in labels_unicos:
        if label <= 1:
            continue

        mask_bolha = np.zeros((h, w), dtype=np.uint8)
        mask_bolha[markers == label] = 255

        cnts, _ = cv2.findContours(mask_bolha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area < 40:
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

        # centroide
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # dentro da ROI circular com folga
        dx = cx - roi_info["cx"]
        dy = cy - roi_info["cy"]
        dist_centro = math.hypot(dx, dy)
        if dist_centro > roi_info["r"] - 5:
            continue

        # medida principal: diâmetro equivalente
        diam_eq_px = math.sqrt((4.0 * area) / math.pi)
        raio_eq_px = diam_eq_px / 2.0

        # filtros físicos
        if diam_eq_px < 8:
            continue
        if circularidade < 0.15:
            continue
        if solidez < 0.35:
            continue

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
            "raio_eq_px": float(raio_eq_px),
            "diam_eq_px": float(diam_eq_px),
            "diam_eq_um": None if diam_eq_um is None else float(diam_eq_um)
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


def desenhar_roi_e_bolhas(img_bgr, roi_info, bolhas, titulo="Bolhas detectadas"):
    out = img_bgr.copy()

    overlay = out.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (roi_info["cx"], roi_info["cy"]), roi_info["r"], 255, -1)
    overlay[mask == 0] = (15, 15, 15)
    out = cv2.addWeighted(out, 0.60, overlay, 0.40, 0)

    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), 2)

    rng = np.random.default_rng(42)
    for i, b in enumerate(bolhas, start=1):
        color = tuple(int(v) for v in rng.integers(60, 256, size=3))
        cv2.drawContours(out, [b["contorno"]], -1, color, 2)
        cv2.putText(
            out,
            str(i),
            (int(round(b["cx"] - 6)), int(round(b["cy"] + 4))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        out,
        f"{titulo}: {len(bolhas)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return out


# ============================================================
# TABELA E GRÁFICOS
# ============================================================
def montar_dataframe_bolhas(bolhas):
    dados = []
    for i, b in enumerate(bolhas, start=1):
        dados.append({
            "Bolha": i,
            "Centro X (px)": round(float(b["cx"]), 1),
            "Centro Y (px)": round(float(b["cy"]), 1),
            "Área (px²)": round(float(b["area_px2"]), 2),
            "Circularidade": round(float(b["circularidade"]), 3),
            "Solidez": round(float(b["solidez"]), 3),
            "Raio equivalente (px)": round(float(b["raio_eq_px"]), 2),
            "Diâmetro equivalente (px)": round(float(b["diam_eq_px"]), 2),
            "Diâmetro equivalente (µm)": None if b["diam_eq_um"] is None else round(float(b["diam_eq_um"]), 2)
        })
    return pd.DataFrame(dados)


def plotar_distribuicao(df):
    if "Diâmetro equivalente (µm)" not in df.columns or df["Diâmetro equivalente (µm)"].dropna().empty:
        return None, None, None

    diametros = df["Diâmetro equivalente (µm)"].dropna().values
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

        marker_factor = st.selectbox(
            "Sensibilidade interna do Watershed",
            [0.20, 0.28, 0.35],
            index=1
        )

        try:
            url_imagem = montar_url_publica(escolhido)
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)

            barra_px_auto, img_calibracao, barra_info = detectar_barra_escala_px(img_original_bgr)

            mask_circular, roi_info = criar_mascara_circular(img_original_bgr.shape, raio_frac=0.43)
            mask_excluir = criar_mascara_regioes_excluidas(img_original_bgr.shape, barra_info)
            mask_roi = combinar_mascaras_roi(mask_circular, mask_excluir)

            prep = preprocessar_base(img_original_bgr, mask_roi)

            ws = detectar_bolhas_watershed(prep["gray_roi"], mask_roi, marker_factor=marker_factor)
            bolhas_finais = extrair_bolhas_dos_markers(
                ws["markers"],
                img_original_bgr.shape,
                roi_info,
                px_per_mm=barra_px_auto
            )

            img_markers = desenhar_markers_coloridos(ws["markers"])
            img_resultado = desenhar_roi_e_bolhas(
                img_original_bgr,
                roi_info,
                bolhas_finais,
                titulo="Bolhas detectadas"
            )

            df = montar_dataframe_bolhas(bolhas_finais)

            st.markdown("## Calibração")
            if barra_px_auto is not None:
                st.success(f"Barra de 1,0 mm detectada automaticamente: {barra_px_auto} px")
            else:
                st.warning("Barra não detectada automaticamente.")

            st.image(cv_to_pil(img_calibracao), caption="Barra de 1,0 mm detectada", width=420)

            st.markdown("## Pré-processamento")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.caption("Imagem original")
                st.image(img_original_pil, width=190)
            with c2:
                st.caption("CLAHE")
                st.image(prep["clahe_vis"], width=190)
            with c3:
                st.caption("Filtro bilateral")
                st.image(prep["bilateral_vis"], width=190)
            with c4:
                st.caption("Sharpen")
                st.image(prep["sharpen_vis"], width=190)

            st.markdown("## Watershed na área útil circular")
            w1, w2, w3 = st.columns(3)
            with w1:
                st.caption("Threshold")
                st.image(ws["thresh"], width=240)
            with w2:
                st.caption("Mapa de distância")
                st.image(ws["dist_norm"], width=240)
            with w3:
                st.caption("Markers")
                st.image(cv_to_pil(img_markers), width=240)

            st.markdown("## Diagnóstico")
            d1, d2 = st.columns(2)
            d1.metric("Marker factor", marker_factor)
            d2.metric("Bolhas detectadas", len(bolhas_finais))

            st.markdown("## Resultado final na área útil circular")
            st.image(cv_to_pil(img_resultado), width=760)

            st.markdown("## Resumo")
            if df.empty:
                st.warning("Nenhuma bolha foi detectada com esta estratégia.")
                return

            st.dataframe(df, use_container_width=True)

            if "Diâmetro equivalente (µm)" in df.columns and not df["Diâmetro equivalente (µm)"].dropna().empty:
                maiores_500 = df[df["Diâmetro equivalente (µm)"] > 500]
                pct_500 = 100.0 * len(maiores_500) / len(df)

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Bolhas totais", len(df))
                s2.metric("Bolhas > 500 µm", len(maiores_500))
                s3.metric("% > 500 µm", f"{pct_500:.2f}%")
                s4.metric("Diâmetro médio (µm)", f"{df['Diâmetro equivalente (µm)'].mean():.2f}")

                st.markdown("## Estatísticas")
                e1, e2 = st.columns(2)
                e1.metric("Mediana (µm)", f"{df['Diâmetro equivalente (µm)'].median():.2f}")
                e2.metric("Máximo (µm)", f"{df['Diâmetro equivalente (µm)'].max():.2f}")

                fig_bar, fig_curve, tabela = plotar_distribuicao(df)
                if fig_bar is not None:
                    st.markdown("## Distribuição granulométrica")
                    st.pyplot(fig_bar)
                    st.markdown("## Curva")
                    st.pyplot(fig_curve)
                    st.markdown("## Quantidade por faixa")
                    st.dataframe(tabela, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
