import io
from typing import Optional, Tuple, List, Dict, Any

import requests
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from openai_bubble_service import (
    analyze_bubbles_with_openai,
    filter_bubbles,
    bubbles_to_rows,
)


# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================
def baixar_imagem(url: str) -> Image.Image:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def baixar_imagem_bytes(url: str) -> Tuple[bytes, str]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    mime_type = response.headers.get("Content-Type", "image/png")
    return response.content, mime_type


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
# DESENHO DOS RESULTADOS
# ============================================================
def desenhar_bolhas(img_bgr: np.ndarray, bubbles: List[Dict[str, Any]]) -> np.ndarray:
    out = img_bgr.copy()
    rng = np.random.default_rng(42)

    for i, b in enumerate(bubbles, start=1):
        color = tuple(int(v) for v in rng.integers(60, 256, size=3))
        x = int(round(float(b["x"])))
        y = int(round(float(b["y"])))
        r = int(round(float(b["radius_px"])))

        cv2.circle(out, (x, y), r, color, 2)
        cv2.putText(
            out,
            str(i),
            (x - 5, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        out,
        f"Bolhas detectadas: {len(bubbles)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return out


# ============================================================
# GRÁFICOS
# ============================================================
def plotar_distribuicao(df: pd.DataFrame):
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
# TELA PRINCIPAL
# ============================================================
def render_consulta_imagens(listar_imagens_supabase, montar_url_publica, session_state):
    with st.container(border=True):
        st.markdown("## Consulta de imagens com OpenAI API")

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

        api_key = st.secrets.get("openai", {}).get("OPENAI_API_KEY", "")
        if not api_key:
            st.warning("Defina OPENAI_API_KEY em st.secrets['openai']['OPENAI_API_KEY'].")
            return

        min_conf = st.slider("Confiança mínima", 0.0, 1.0, 0.35, 0.05)
        min_radius = st.slider("Raio mínimo (px)", 1.0, 20.0, 3.0, 0.5)
        model_name = st.text_input("Modelo", value="gpt-5.4")

        if not st.button("Analisar imagem com OpenAI", key="btn_openai_analisar"):
            return

        try:
            url_imagem = montar_url_publica(escolhido)

            img_pil = baixar_imagem(url_imagem)
            img_bgr = pil_to_cv(img_pil)

            img_bytes, mime_type = baixar_imagem_bytes(url_imagem)

            barra_px_auto, img_calibracao, _ = detectar_barra_escala_px(img_bgr)

            with st.spinner("Enviando imagem para a OpenAI API..."):
                result = analyze_bubbles_with_openai(
                    api_key=api_key,
                    image_bytes=img_bytes,
                    mime_type=mime_type,
                    model=model_name,
                )

            bubbles_raw = result.get("bubbles", [])
            scale_bar_px_model = result.get("scale_bar_px", None)

            bubbles = filter_bubbles(
                bubbles_raw,
                min_radius_px=min_radius,
                min_confidence=min_conf,
            )

            # prioridade para a régua detectada localmente; se falhar, usa a da IA
            px_per_mm = barra_px_auto if barra_px_auto else scale_bar_px_model

            rows = bubbles_to_rows(bubbles, px_per_mm=px_per_mm)
            df = pd.DataFrame(rows)

            img_marked = desenhar_bolhas(img_bgr, bubbles)

            st.markdown("## Calibração")
            if barra_px_auto:
                st.success(f"Barra detectada localmente: {barra_px_auto:.2f} px")
            elif scale_bar_px_model:
                st.info(f"Barra estimada pela IA: {float(scale_bar_px_model):.2f} px")
            else:
                st.warning("Não foi possível calibrar a escala automaticamente.")

            st.image(cv_to_pil(img_calibracao), caption="Barra de 1,0 mm", width=420)

            st.markdown("## Resultado")
            st.image(cv_to_pil(img_marked), width=900)

            st.markdown("## Resumo textual da IA")
            st.write(result.get("image_summary", ""))

            st.markdown("## Diagnóstico")
            c1, c2 = st.columns(2)
            c1.metric("Bolhas retornadas pela IA", len(bubbles_raw))
            c2.metric("Bolhas após filtros", len(bubbles))

            if df.empty:
                st.warning("A IA não retornou bolhas válidas com os filtros atuais.")
                st.code(json.dumps(result, indent=2, ensure_ascii=False))
                return

            st.markdown("## Tabela")
            st.dataframe(df, use_container_width=True)

            if "Diâmetro (µm)" in df.columns and not df["Diâmetro (µm)"].dropna().empty:
                maiores_500 = df[df["Diâmetro (µm)"] > 500]
                pct_500 = 100.0 * len(maiores_500) / len(df)

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Bolhas totais", len(df))
                s2.metric("Bolhas > 500 µm", len(maiores_500))
                s3.metric("% > 500 µm", f"{pct_500:.2f}%")
                s4.metric("Diâmetro médio (µm)", f"{df['Diâmetro (µm)'].mean():.2f}")

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
            st.error(f"Erro ao analisar a imagem: {e}")
