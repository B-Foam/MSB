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
# ROI CIRCULAR
# ============================================================
def criar_mascara_circular(shape, raio_frac=0.43):
    h, w = shape[:2]
    cx = w // 2
    cy = h // 2

    r = int(min(h, w) * raio_frac)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    return mask, {"cx": cx, "cy": cy, "r": r}


def desenhar_roi_circular(img_bgr, roi_info):
    out = img_bgr.copy()
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 0), 2)
    return out


# ============================================================
# MÁSCARAS EXTRAS
# ============================================================
def criar_mascara_regioes_excluidas(shape, barra_info=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # faixa superior com texto
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
    # região útil = círculo E não excluído
    roi = np.zeros_like(mask_circular)
    roi[(mask_circular == 255) & (mask_excluir == 0)] = 255
    return roi


def aplicar_roi(img_gray, mask_roi, valor_fora=0):
    out = img_gray.copy()
    out[mask_roi == 0] = valor_fora
    return out


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================
def preprocessar_base(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    bilateral = cv2.bilateralFilter(clahe_img, 9, 75, 75)

    # realce suave
    blur_ref = cv2.GaussianBlur(bilateral, (0, 0), 1.2)
    sharpen = cv2.addWeighted(bilateral, 1.40, blur_ref, -0.40, 0)

    # LoG absoluto
    log_img = cv2.Laplacian(sharpen, cv2.CV_32F, ksize=3)
    log_abs = np.abs(log_img)
    log_abs = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # aplicar ROI só no processamento
    gray_proc = aplicar_roi(gray, mask_roi, valor_fora=0)
    clahe_proc = aplicar_roi(clahe_img, mask_roi, valor_fora=0)
    bilateral_proc = aplicar_roi(bilateral, mask_roi, valor_fora=0)
    sharpen_proc = aplicar_roi(sharpen, mask_roi, valor_fora=0)
    log_proc = aplicar_roi(log_abs, mask_roi, valor_fora=0)

    return {
        "gray_vis": gray,
        "clahe_vis": clahe_img,
        "bilateral_vis": bilateral,
        "sharpen_vis": sharpen,
        "log_vis": log_abs,
        "gray_proc": gray_proc,
        "clahe_proc": clahe_proc,
        "bilateral_proc": bilateral_proc,
        "sharpen_proc": sharpen_proc,
        "log_proc": log_proc
    }


# ============================================================
# DETECÇÃO BLOB MULTIESCALA VIA LoG
# ============================================================
def obter_sigmas(px_per_mm=None):
    if px_per_mm is not None and px_per_mm > 0:
        # aproximadamente da ordem dos raios observados
        sigmas = [
            px_per_mm * 0.010,
            px_per_mm * 0.015,
            px_per_mm * 0.022,
            px_per_mm * 0.032,
            px_per_mm * 0.045,
            px_per_mm * 0.065,
            px_per_mm * 0.090
        ]
    else:
        sigmas = [2.0, 3.0, 4.5, 6.0, 8.0, 11.0, 15.0]

    sigmas = [float(max(1.5, s)) for s in sigmas]
    return sigmas


def detectar_blobs_log_multiescala(img_gray, mask_roi, px_per_mm=None):
    sigmas = obter_sigmas(px_per_mm)
    candidatos = []
    h, w = img_gray.shape[:2]

    for sigma in sigmas:
        blur = cv2.GaussianBlur(img_gray, (0, 0), sigma)
        log = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        resp = np.abs(log)
        resp = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)

        # threshold local suave por escala
        thr_val = max(18, np.percentile(resp[mask_roi == 255], 92))
        pts = np.argwhere(resp >= thr_val)

        step = max(1, len(pts) // 2500)
        pts = pts[::step]

        for y, x in pts:
            if mask_roi[y, x] == 0:
                continue

            r = math.sqrt(2.0) * sigma
            if r < 4 or r > 120:
                continue

            candidatos.append({
                "x": int(x),
                "y": int(y),
                "r": float(r),
                "sigma": float(sigma),
                "resp": float(resp[y, x])
            })

    return candidatos


# ============================================================
# VALIDAÇÃO DOS BLOBs
# ============================================================
def media_anel(img, cx, cy, r, esp=1.5, n=72):
    h, w = img.shape[:2]
    vals = []

    for rr in [max(1.0, r - esp), r, r + esp]:
        for k in range(n):
            a = 2 * np.pi * k / n
            x = int(round(cx + rr * np.cos(a)))
            y = int(round(cy + rr * np.sin(a)))
            if 0 <= x < w and 0 <= y < h:
                vals.append(float(img[y, x]))

    if len(vals) == 0:
        return 0.0, 0.0

    vals = np.array(vals, dtype=np.float32)
    return float(np.mean(vals)), float(np.std(vals))


def media_disco(img, cx, cy, r):
    h, w = img.shape[:2]
    x0 = max(0, int(cx - r))
    x1 = min(w, int(cx + r + 1))
    y0 = max(0, int(cy - r))
    y1 = min(h, int(cy + r + 1))

    if x1 <= x0 or y1 <= y0:
        return 0.0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    vals = img[y0:y1, x0:x1][mask]

    if vals.size == 0:
        return 0.0

    return float(np.mean(vals))


def dentro_roi_circular(x, y, roi_info, folga=0):
    dx = x - roi_info["cx"]
    dy = y - roi_info["cy"]
    return (dx * dx + dy * dy) <= (roi_info["r"] - folga) ** 2


def validar_blob(cand, prep, roi_info, mask_roi):
    x = cand["x"]
    y = cand["y"]
    r = cand["r"]

    if not dentro_roi_circular(x, y, roi_info, folga=int(r + 3)):
        return None

    if mask_roi[y, x] == 0:
        return None

    gray = prep["gray_proc"]
    log_img = prep["log_proc"]
    sharp = prep["sharpen_proc"]

    # medidas internas / anel / externas
    inside = media_disco(gray, x, y, max(1, int(r * 0.55)))
    ring_mean_gray, ring_std_gray = media_anel(gray, x, y, r, esp=max(1.0, r * 0.08), n=96)
    outer = media_disco(gray, x, y, max(1, int(r * 1.25))) - media_disco(gray, x, y, max(1, int(r * 0.95)))

    ring_mean_log, ring_std_log = media_anel(log_img, x, y, r, esp=max(1.0, r * 0.08), n=96)
    ring_mean_sharp, ring_std_sharp = media_anel(sharp, x, y, r, esp=max(1.0, r * 0.08), n=96)

    contrast_ring = abs(ring_mean_gray - inside)
    log_strength = ring_mean_log
    stability = 1.0 / (1.0 + ring_std_log / max(1.0, ring_mean_log))

    # score de validação
    score = (
        1.8 * cand["resp"] +
        2.0 * log_strength +
        5.0 * contrast_ring +
        120.0 * stability +
        1.2 * r
    )

    # filtros mínimos
    if log_strength < 12:
        return None
    if contrast_ring < 1.2:
        return None
    if stability < 0.12:
        return None

    out = cand.copy()
    out.update({
        "score": float(score),
        "contrast_ring": float(contrast_ring),
        "log_strength": float(log_strength),
        "stability": float(stability)
    })
    return out


def validar_blobs(candidatos, prep, roi_info, mask_roi):
    validos = []
    for c in candidatos:
        v = validar_blob(c, prep, roi_info, mask_roi)
        if v is not None:
            validos.append(v)
    return validos


# ============================================================
# REMOÇÃO DE DUPLICATAS
# ============================================================
def agrupar_candidatos(candidatos):
    grupos = []
    usados = [False] * len(candidatos)

    for i, ci in enumerate(candidatos):
        if usados[i]:
            continue

        grupo = [i]
        usados[i] = True
        mudou = True

        while mudou:
            mudou = False
            for j, cj in enumerate(candidatos):
                if usados[j]:
                    continue

                for idx in grupo:
                    ck = candidatos[idx]
                    dist = math.hypot(ck["x"] - cj["x"], ck["y"] - cj["y"])
                    r_ref = max(ck["r"], cj["r"])
                    dr = abs(ck["r"] - cj["r"])

                    if dist <= 0.9 * r_ref and dr <= 0.6 * r_ref:
                        grupo.append(j)
                        usados[j] = True
                        mudou = True
                        break

        grupos.append([candidatos[k] for k in grupo])

    return grupos


def manter_melhor_por_grupo(candidatos):
    if not candidatos:
        return []

    grupos = agrupar_candidatos(candidatos)
    finais = []

    for g in grupos:
        melhor = max(g, key=lambda x: x["score"])
        finais.append(melhor)

    return finais


# ============================================================
# MEDIÇÃO
# ============================================================
def montar_dataframe_bolhas(candidatos, px_per_mm):
    dados = []
    for i, c in enumerate(candidatos, start=1):
        diam_px = 2.0 * c["r"]
        diam_um = None

        if px_per_mm is not None and px_per_mm > 0:
            diam_um = (diam_px / px_per_mm) * 1000.0

        dados.append({
            "Bolha": i,
            "Centro X (px)": c["x"],
            "Centro Y (px)": c["y"],
            "Raio (px)": round(float(c["r"]), 2),
            "Diâmetro (px)": round(float(diam_px), 2),
            "Diâmetro (µm)": None if diam_um is None else round(float(diam_um), 2),
            "Score": round(float(c["score"]), 2),
            "Contraste do anel": round(float(c["contrast_ring"]), 2),
            "Força LoG": round(float(c["log_strength"]), 2),
            "Estabilidade": round(float(c["stability"]), 3)
        })

    return pd.DataFrame(dados)


def plotar_distribuicao(df):
    if "Diâmetro (µm)" not in df.columns or df["Diâmetro (µm)"].dropna().empty:
        return None, None, None

    diametros = df["Diâmetro (µm)"].dropna().values
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
# DESENHO
# ============================================================
def desenhar_roi_e_bolhas(img_bgr, roi_info, bolhas, titulo="Bolhas detectadas"):
    out = img_bgr.copy()

    # sombrear fora da ROI
    overlay = out.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (roi_info["cx"], roi_info["cy"]), roi_info["r"], 255, -1)
    overlay[mask == 0] = (15, 15, 15)
    out = cv2.addWeighted(out, 0.60, overlay, 0.40, 0)

    # círculo da ROI
    cv2.circle(out, (roi_info["cx"], roi_info["cy"]), roi_info["r"], (255, 255, 255), 2)

    # bolhas
    rng = np.random.default_rng(42)
    for i, b in enumerate(bolhas, start=1):
        color = tuple(int(v) for v in rng.integers(60, 256, size=3))
        cv2.circle(out, (int(b["x"]), int(b["y"])), int(round(b["r"])), color, 2)
        cv2.putText(
            out,
            str(i),
            (int(b["x"] - 6), int(b["y"] + 4)),
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

        try:
            url_imagem = montar_url_publica(escolhido)
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)

            barra_px_auto, img_calibracao, barra_info = detectar_barra_escala_px(img_original_bgr)

            mask_circular, roi_info = criar_mascara_circular(img_original_bgr.shape, raio_frac=0.43)
            mask_excluir = criar_mascara_regioes_excluidas(img_original_bgr.shape, barra_info)
            mask_roi = combinar_mascaras_roi(mask_circular, mask_excluir)

            prep = preprocessar_base(img_original_bgr, mask_roi)

            candidatos_brutos = detectar_blobs_log_multiescala(
                prep["log_proc"],
                mask_roi,
                px_per_mm=barra_px_auto
            )

            candidatos_validos = validar_blobs(
                candidatos_brutos,
                prep,
                roi_info,
                mask_roi
            )

            bolhas_finais = manter_melhor_por_grupo(candidatos_validos)

            img_resultado = desenhar_roi_e_bolhas(
                img_original_bgr,
                roi_info,
                bolhas_finais,
                titulo="Bolhas detectadas"
            )

            df = montar_dataframe_bolhas(bolhas_finais, barra_px_auto)

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
                st.caption("LoG absoluto")
                st.image(prep["log_vis"], width=190)

            st.markdown("## Diagnóstico")
            d1, d2, d3 = st.columns(3)
            d1.metric("Candidatos brutos", len(candidatos_brutos))
            d2.metric("Candidatos válidos", len(candidatos_validos))
            d3.metric("Bolhas detectadas", len(bolhas_finais))

            st.markdown("## Resultado final na área útil circular")
            st.image(cv_to_pil(img_resultado), width=760)

            st.markdown("## Resumo")
            if df.empty:
                st.warning("Nenhuma bolha foi detectada com a estratégia da ROI circular.")
                return

            st.dataframe(df, use_container_width=True)

            if "Diâmetro (µm)" in df.columns and not df["Diâmetro (µm)"].dropna().empty:
                maiores_500 = df[df["Diâmetro (µm)"] > 500]
                pct_500 = 100.0 * len(maiores_500) / len(df)

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Bolhas totais", len(df))
                s2.metric("Bolhas > 500 µm", len(maiores_500))
                s3.metric("% > 500 µm", f"{pct_500:.2f}%")
                s4.metric("Diâmetro médio (µm)", f"{df['Diâmetro (µm)'].mean():.2f}")

                st.markdown("## Estatísticas")
                e1, e2 = st.columns(2)
                e1.metric("Mediana (µm)", f"{df['Diâmetro (µm)'].median():.2f}")
                e2.metric("Máximo (µm)", f"{df['Diâmetro (µm)'].max():.2f}")

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
