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
# REGIÕES EXCLUÍDAS
# ============================================================
def criar_mascara_regioes_excluidas(shape, barra_info=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    mask[0:int(h * 0.035), :] = 255

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


def aplicar_mascara_exclusao(img_gray: np.ndarray, mask_exclusao: np.ndarray, valor: int = 255):
    out = img_gray.copy()
    out[mask_exclusao == 255] = valor
    return out


# ============================================================
# PRÉ-PROCESSAMENTO NOVO
# ============================================================
def preprocessar_base(img_bgr: np.ndarray, mask_exclusao: np.ndarray):
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    clahe_full = clahe.apply(gray_full)

    bilateral_full = cv2.bilateralFilter(clahe_full, 9, 75, 75)

    blur_ref = cv2.GaussianBlur(bilateral_full, (0, 0), 1.2)
    sharpen = cv2.addWeighted(bilateral_full, 1.45, blur_ref, -0.45, 0)

    log_img = cv2.Laplacian(sharpen, cv2.CV_32F, ksize=3)
    log_abs = np.abs(log_img)
    log_abs = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    canny = cv2.Canny(sharpen, 40, 110)

    gray_vis = gray_full.copy()
    clahe_vis = clahe_full.copy()
    bilateral_vis = bilateral_full.copy()
    sharpen_vis = sharpen.copy()
    log_vis = log_abs.copy()
    canny_vis = canny.copy()

    bilateral_proc = aplicar_mascara_exclusao(bilateral_full, mask_exclusao, valor=255)
    sharpen_proc = aplicar_mascara_exclusao(sharpen, mask_exclusao, valor=255)
    log_proc = aplicar_mascara_exclusao(log_abs, mask_exclusao, valor=0)
    canny_proc = aplicar_mascara_exclusao(canny, mask_exclusao, valor=0)

    return {
        "gray_vis": gray_vis,
        "clahe_vis": clahe_vis,
        "bilateral_vis": bilateral_vis,
        "sharpen_vis": sharpen_vis,
        "log_vis": log_vis,
        "canny_vis": canny_vis,
        "bilateral_proc": bilateral_proc,
        "sharpen_proc": sharpen_proc,
        "log_proc": log_proc,
        "canny_proc": canny_proc
    }


# ============================================================
# FAIXAS AUTOMÁTICAS DE RAIO
# ============================================================
def obter_faixas_raio(shape, px_per_mm=None):
    h, w = shape[:2]
    base = min(h, w)

    if px_per_mm is not None and px_per_mm > 0:
        pequeno = (max(4, int(px_per_mm * 0.012)), max(8, int(px_per_mm * 0.045)))
        medio = (max(8, int(px_per_mm * 0.045)), max(18, int(px_per_mm * 0.12)))
        grande = (max(18, int(px_per_mm * 0.12)), max(50, int(px_per_mm * 0.28)))
    else:
        pequeno = (4, max(10, int(base * 0.012)))
        medio = (max(10, int(base * 0.012)), max(24, int(base * 0.035)))
        grande = (max(24, int(base * 0.035)), max(55, int(base * 0.09)))

    return [
        ("Pequenas", pequeno[0], pequeno[1], 12),
        ("Médias", medio[0], medio[1], 18),
        ("Grandes", grande[0], grande[1], 28),
    ]


# ============================================================
# Hough em múltiplos canais
# ============================================================
def detectar_hough_em_canal(img_gray, nome_canal, faixas):
    candidatos = []

    for nome_faixa, rmin, rmax, min_dist in faixas:
        if nome_faixa == "Pequenas":
            dp, p1, p2 = 1.1, 95, 11
        elif nome_faixa == "Médias":
            dp, p1, p2 = 1.15, 100, 14
        else:
            dp, p1, p2 = 1.2, 110, 16

        circles = cv2.HoughCircles(
            img_gray,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=p1,
            param2=p2,
            minRadius=rmin,
            maxRadius=rmax
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for c in circles:
                candidatos.append({
                    "x": int(c[0]),
                    "y": int(c[1]),
                    "r": int(c[2]),
                    "faixa": nome_faixa,
                    "canal": nome_canal
                })

    return candidatos


def detectar_candidatos_multicanal(prep, faixas):
    c1 = detectar_hough_em_canal(prep["bilateral_proc"], "Bilateral", faixas)
    c2 = detectar_hough_em_canal(prep["sharpen_proc"], "Realçada", faixas)
    c3 = detectar_hough_em_canal(prep["log_proc"], "LoG", faixas)

    return c1 + c2 + c3


# ============================================================
# AJUDAS GEOMÉTRICAS
# ============================================================
def ponto_em_regiao_util(x, y, shape, margem_x_frac=0.05, margem_y_frac=0.04):
    h, w = shape[:2]
    mx = int(w * margem_x_frac)
    my = int(h * margem_y_frac)
    return (mx <= x <= w - mx) and (my <= y <= h - my)


def circulo_longe_da_borda(x, y, r, shape, margem=3):
    h, w = shape[:2]
    return (
        x - r > margem and
        y - r > margem and
        x + r < w - margem and
        y + r < h - margem
    )


def sample_circle_points(cx, cy, r, n=72):
    angs = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = []
    for a in angs:
        x = int(round(cx + r * np.cos(a)))
        y = int(round(cy + r * np.sin(a)))
        pts.append((x, y))
    return pts


def valores_em_circulo(img, cx, cy, r, n=72):
    h, w = img.shape[:2]
    vals = []
    for x, y in sample_circle_points(cx, cy, r, n=n):
        if 0 <= x < w and 0 <= y < h:
            vals.append(float(img[y, x]))
    if len(vals) == 0:
        return np.array([0.0], dtype=np.float32)
    return np.array(vals, dtype=np.float32)


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


# ============================================================
# VALIDAÇÃO MAIS LEVE E MELHOR
# ============================================================
def validar_candidato_bolha(cand, gray_img, edge_img, shape):
    cx, cy, r = cand["x"], cand["y"], cand["r"]

    if not ponto_em_regiao_util(cx, cy, shape, 0.05, 0.04):
        return None
    if not circulo_longe_da_borda(cx, cy, r, shape, margem=3):
        return None
    if r < 4:
        return None

    ring_in = valores_em_circulo(edge_img, cx, cy, max(2, int(round(r * 0.90))), n=96)
    ring_mid = valores_em_circulo(edge_img, cx, cy, r, n=96)
    ring_out = valores_em_circulo(edge_img, cx, cy, max(2, int(round(r * 1.10))), n=96)

    edge_vals = np.concatenate([ring_in, ring_mid, ring_out])
    edge_mean = float(np.mean(edge_vals))
    edge_std = float(np.std(edge_vals))

    thr = max(14.0, edge_mean * 0.85)
    edge_fraction = float(np.mean(edge_vals > thr))

    inside_mean = media_disco(gray_img, cx, cy, max(1, int(round(r * 0.55))))
    ring_gray = float(np.mean(valores_em_circulo(gray_img, cx, cy, r, n=96)))

    out1 = media_disco(gray_img, cx, cy, max(1, int(round(r * 1.35))))
    out2 = media_disco(gray_img, cx, cy, max(1, int(round(r * 1.10))))
    outside_band = out1 - out2

    contrast_ring = abs(ring_gray - inside_mean)
    stability = 1.0 / (1.0 + edge_std / max(1.0, edge_mean))

    # bônus por canal
    canal_bonus = 0.0
    if cand["canal"] == "Realçada":
        canal_bonus = 12.0
    elif cand["canal"] == "LoG":
        canal_bonus = 8.0
    else:
        canal_bonus = 10.0

    score = (
        0.55 * edge_mean +
        140.0 * edge_fraction +
        0.45 * contrast_ring +
        90.0 * stability +
        0.22 * r +
        canal_bonus
    )

    # filtros mínimos mais brandos que a versão anterior
    if edge_fraction < 0.12:
        return None
    if edge_mean < 10:
        return None
    if contrast_ring < 1.5:
        return None

    out = cand.copy()
    out.update({
        "score": float(score),
        "edge_mean": float(edge_mean),
        "edge_fraction": float(edge_fraction),
        "contrast_ring": float(contrast_ring),
        "stability": float(stability),
        "outside_band": float(outside_band)
    })
    return out


def validar_candidatos(candidatos, gray_img, edge_img, shape):
    validos = []
    for c in candidatos:
        v = validar_candidato_bolha(c, gray_img, edge_img, shape)
        if v is not None:
            validos.append(v)
    return validos


# ============================================================
# REMOÇÃO DE DUPLICATAS
# ============================================================
def agrupar_candidatos_proximos(candidatos):
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

                    if dist <= 0.85 * r_ref and dr <= 0.55 * r_ref:
                        grupo.append(j)
                        usados[j] = True
                        mudou = True
                        break

        grupos.append([candidatos[k] for k in grupo])

    return grupos


def manter_melhor_por_grupo(candidatos):
    if not candidatos:
        return []

    grupos = agrupar_candidatos_proximos(candidatos)
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
            "Continuidade": round(float(c["edge_fraction"]), 3),
            "Contraste do anel": round(float(c["contrast_ring"]), 2),
            "Canal": c["canal"],
            "Faixa": c["faixa"]
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
def desenhar_candidatos(img_bgr, candidatos, titulo="Bolhas detectadas"):
    out = img_bgr.copy()
    rng = np.random.default_rng(42)

    for c in candidatos:
        color = tuple(int(v) for v in rng.integers(60, 256, size=3))
        cv2.circle(out, (int(c["x"]), int(c["y"])), int(c["r"]), color, 2)

    cv2.putText(
        out,
        f"{titulo}: {len(candidatos)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return out


def desenhar_regiao_util(img_bgr):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    mx = int(w * 0.05)
    my = int(h * 0.04)
    cv2.rectangle(out, (mx, my), (w - mx, h - my), (255, 255, 0), 2)
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
            mask_exclusao = criar_mascara_regioes_excluidas(img_original_bgr.shape, barra_info)

            prep = preprocessar_base(img_original_bgr, mask_exclusao)
            faixas = obter_faixas_raio(img_original_bgr.shape, barra_px_auto)

            candidatos_brutos = detectar_candidatos_multicanal(prep, faixas)
            candidatos_validos = validar_candidatos(
                candidatos_brutos,
                prep["gray_vis"],
                prep["canny_vis"],
                img_original_bgr.shape
            )
            candidatos_finais = manter_melhor_por_grupo(candidatos_validos)

            img_regiao = desenhar_regiao_util(img_original_bgr)
            img_brutos = desenhar_candidatos(img_original_bgr, candidatos_brutos, "Candidatos brutos")
            img_validos = desenhar_candidatos(img_original_bgr, candidatos_validos, "Após validação radial")
            img_finais = desenhar_candidatos(img_original_bgr, candidatos_finais, "Bolhas detectadas")

            df = montar_dataframe_bolhas(candidatos_finais, barra_px_auto)

            st.markdown("## Calibração")
            if barra_px_auto is not None:
                st.success(f"Barra de 1,0 mm detectada automaticamente: {barra_px_auto} px")
            else:
                st.warning("Barra não detectada automaticamente.")

            st.image(cv_to_pil(img_calibracao), caption="Barra de 1,0 mm detectada", width=460)

            st.markdown("## Pré-processamento")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.caption("Imagem original")
                st.image(img_original_pil, width=210)
            with c2:
                st.caption("CLAHE")
                st.image(prep["clahe_vis"], width=210)
            with c3:
                st.caption("Filtro bilateral")
                st.image(prep["bilateral_vis"], width=210)
            with c4:
                st.caption("Realce suave")
                st.image(prep["sharpen_vis"], width=210)

            st.markdown("## Diagnóstico")
            d1, d2 = st.columns(2)
            with d1:
                st.caption("LoG absoluto")
                st.image(prep["log_vis"], width=360)
            with d2:
                st.caption("Região útil considerada")
                st.image(cv_to_pil(img_regiao), width=360)

            st.markdown("## Nova estratégia")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.caption("Candidatos brutos multicanal")
                st.image(cv_to_pil(img_brutos), width=260)
            with r2:
                st.caption("Após validação radial")
                st.image(cv_to_pil(img_validos), width=260)
            with r3:
                st.caption("Após remover duplicatas")
                st.image(cv_to_pil(img_finais), width=260)

            st.markdown("## Diagnóstico numérico")
            k1, k2, k3 = st.columns(3)
            k1.metric("Candidatos brutos", len(candidatos_brutos))
            k2.metric("Candidatos válidos", len(candidatos_validos))
            k3.metric("Bolhas detectadas", len(candidatos_finais))

            st.markdown("## Resumo")
            if df.empty:
                st.warning("Nenhuma bolha foi detectada com a nova estratégia.")
                return

            st.dataframe(df, use_container_width=True)

            if "Diâmetro (µm)" in df.columns and not df["Diâmetro (µm)"].dropna().empty:
                maiores_500 = df[df["Diâmetro (µm)"] > 500]
                pct_500 = 100.0 * len(maiores_500) / len(df)

                s1, s2, s3 = st.columns(3)
                s1.metric("Bolhas > 500 µm", len(maiores_500))
                s2.metric("% > 500 µm", f"{pct_500:.2f}%")
                s3.metric("Diâmetro médio (µm)", f"{df['Diâmetro (µm)'].mean():.2f}")

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
