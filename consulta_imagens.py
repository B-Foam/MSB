import io
import math
from typing import Dict, List, Tuple, Any, Optional

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
# ESTADO
# ============================================================
def garantir_estado(session_state):
    if "lista_imagens_consulta" not in session_state:
        session_state.lista_imagens_consulta = []

    if "circles_by_image" not in session_state:
        session_state.circles_by_image = {}

    if "roi_by_image" not in session_state:
        session_state.roi_by_image = {}


def obter_circulos(session_state, image_name: str) -> List[Dict[str, Any]]:
    garantir_estado(session_state)
    return session_state.circles_by_image.get(image_name, [])


def salvar_circulos(session_state, image_name: str, circles: List[Dict[str, Any]]):
    garantir_estado(session_state)
    session_state.circles_by_image[image_name] = circles


def obter_roi(session_state, image_name: str) -> Optional[Dict[str, Any]]:
    garantir_estado(session_state)
    return session_state.roi_by_image.get(image_name)


def salvar_roi(session_state, image_name: str, roi_info: Dict[str, Any]):
    garantir_estado(session_state)
    session_state.roi_by_image[image_name] = roi_info


# ============================================================
# CALIBRAÇÃO DA BARRA
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
    return float(ww), img_annot, barra_info


# ============================================================
# ROI CIRCULAR
# ============================================================
def criar_roi_circular_padrao(shape, raio_frac=0.43):
    h, w = shape[:2]
    cx = w // 2
    cy = h // 2
    r = int(min(h, w) * raio_frac)
    return {"cx": cx, "cy": cy, "r": r}


def criar_mascara_roi(shape, roi_info: Dict[str, Any], barra_info=None):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.circle(mask, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 255, -1)

    # excluir cabeçalho superior
    mask[0:int(h * 0.035), :] = 0

    # excluir barra de escala
    if barra_info is not None:
        x = barra_info["x"]
        y = barra_info["y"]
        ww = barra_info["w"]
        hh = barra_info["h"]

        x_ini = max(0, x - 25)
        y_ini = max(0, y - 60)
        x_fim = min(w, x + ww + 90)
        y_fim = min(h, y + hh + 20)

        mask[y_ini:y_fim, x_ini:x_fim] = 0

    return mask


def ponto_dentro_roi(x: float, y: float, roi_info: Dict[str, Any], folga: float = 0.0) -> bool:
    dx = x - roi_info["cx"]
    dy = y - roi_info["cy"]
    return (dx * dx + dy * dy) <= (roi_info["r"] - folga) ** 2


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================
def preprocessar_para_candidatos(img_bgr: np.ndarray, mask_roi: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    bilateral = cv2.bilateralFilter(clahe_img, 9, 75, 75)

    blur_ref = cv2.GaussianBlur(bilateral, (0, 0), 1.2)
    sharpen = cv2.addWeighted(bilateral, 1.40, blur_ref, -0.40, 0)

    roi_gray = gray.copy()
    roi_gray[mask_roi == 0] = 0

    roi_clahe = clahe_img.copy()
    roi_clahe[mask_roi == 0] = 0

    roi_sharpen = sharpen.copy()
    roi_sharpen[mask_roi == 0] = 0

    return {
        "gray": gray,
        "clahe": clahe_img,
        "sharpen": sharpen,
        "roi_gray": roi_gray,
        "roi_clahe": roi_clahe,
        "roi_sharpen": roi_sharpen,
    }


# ============================================================
# CANDIDATOS INICIAIS
# ============================================================
def gerar_candidatos_iniciais(
    img_proc: np.ndarray,
    roi_info: Dict[str, Any],
    px_per_mm: Optional[float] = None,
) -> List[Dict[str, Any]]:
    candidatos = []

    if px_per_mm and px_per_mm > 0:
        min_r = max(4, int(px_per_mm * 0.010))
        max_r = max(12, int(px_per_mm * 0.18))
    else:
        min_r = 4
        max_r = 60

    circles = cv2.HoughCircles(
        img_proc,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(8, min_r * 2),
        param1=100,
        param2=16,
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for c in circles:
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            if ponto_dentro_roi(x, y, roi_info, folga=r + 2):
                candidatos.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "r": float(r),
                        "ativo": True,
                    }
                )

    return remover_duplicados(candidatos)


def remover_duplicados(circles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not circles:
        return []

    finais = []
    for c in circles:
        manter = True
        for f in finais:
            dist = math.hypot(c["x"] - f["x"], c["y"] - f["y"])
            r_ref = max(c["r"], f["r"])
            if dist <= 0.8 * r_ref and abs(c["r"] - f["r"]) <= 0.6 * r_ref:
                manter = False
                break
        if manter:
            finais.append(c)
    return finais


# ============================================================
# EDIÇÃO DOS CÍRCULOS
# ============================================================
def circles_to_dataframe(circles: List[Dict[str, Any]]) -> pd.DataFrame:
    if not circles:
        return pd.DataFrame(columns=["ativo", "x", "y", "r"])
    return pd.DataFrame(circles)[["ativo", "x", "y", "r"]]


def dataframe_to_circles(df: pd.DataFrame, roi_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    circles = []
    for _, row in df.iterrows():
        try:
            ativo = bool(row["ativo"])
            x = float(row["x"])
            y = float(row["y"])
            r = float(row["r"])
        except Exception:
            continue

        if r <= 0:
            continue
        if not ponto_dentro_roi(x, y, roi_info, folga=r + 1):
            continue

        circles.append({
            "ativo": ativo,
            "x": x,
            "y": y,
            "r": r,
        })

    return remover_duplicados(circles)


# ============================================================
# DESENHO
# ============================================================
def desenhar_roi_e_circulos(
    img_bgr: np.ndarray,
    roi_info: Dict[str, Any],
    circles: List[Dict[str, Any]],
    titulo: str = "Bolhas"
) -> np.ndarray:
    out = img_bgr.copy()

    overlay = out.copy()
    h, w = out.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(roi_info["cx"]), int(roi_info["cy"])), int(roi_info["r"]), 255, -1)
    overlay[mask == 0] = (15, 15, 15)
    out = cv2.addWeighted(out, 0.65, overlay, 0.35, 0)

    cv2.circle(
        out,
        (int(roi_info["cx"]), int(roi_info["cy"])),
        int(roi_info["r"]),
        (255, 255, 255),
        2
    )

    rng = np.random.default_rng(42)
    count_ativos = 0

    for i, c in enumerate(circles, start=1):
        if not c.get("ativo", True):
            continue

        count_ativos += 1
        color = tuple(int(v) for v in rng.integers(60, 256, size=3))
        x = int(round(c["x"]))
        y = int(round(c["y"]))
        r = int(round(c["r"]))

        cv2.circle(out, (x, y), r, color, 2)
        cv2.putText(
            out,
            str(count_ativos),
            (x - 5, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        out,
        f"{titulo}: {count_ativos}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return out


# ============================================================
# MEDIÇÃO
# ============================================================
def montar_dataframe_medidas(circles: List[Dict[str, Any]], px_per_mm: Optional[float]) -> pd.DataFrame:
    rows = []
    idx = 1
    for c in circles:
        if not c.get("ativo", True):
            continue

        diam_px = 2.0 * float(c["r"])
        diam_um = None
        if px_per_mm and px_per_mm > 0:
            diam_um = (diam_px / px_per_mm) * 1000.0

        rows.append({
            "Bolha": idx,
            "Centro X (px)": round(float(c["x"]), 1),
            "Centro Y (px)": round(float(c["y"]), 1),
            "Raio (px)": round(float(c["r"]), 2),
            "Diâmetro (px)": round(float(diam_px), 2),
            "Diâmetro (µm)": None if diam_um is None else round(float(diam_um), 2),
        })
        idx += 1

    return pd.DataFrame(rows)


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
    garantir_estado(session_state)

    with st.container(border=True):
        st.markdown("## Consulta de imagens sem API paga")

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

        url_imagem = montar_url_publica(escolhido)
        img_pil = baixar_imagem(url_imagem)
        img_bgr = pil_to_cv(img_pil)

        px_per_mm, img_calibracao, barra_info = detectar_barra_escala_px(img_bgr)

        roi_info = obter_roi(session_state, escolhido)
        if roi_info is None:
            roi_info = criar_roi_circular_padrao(img_bgr.shape, raio_frac=0.43)
            salvar_roi(session_state, escolhido, roi_info)

        # ---------------- ROI ----------------
        st.markdown("## Área útil circular")

        r1, r2, r3 = st.columns(3)
        with r1:
            roi_cx = st.number_input("Centro X ROI", min_value=0, max_value=int(img_bgr.shape[1]), value=int(roi_info["cx"]), step=1)
        with r2:
            roi_cy = st.number_input("Centro Y ROI", min_value=0, max_value=int(img_bgr.shape[0]), value=int(roi_info["cy"]), step=1)
        with r3:
            roi_r = st.number_input("Raio ROI", min_value=10, max_value=int(min(img_bgr.shape[:2])), value=int(roi_info["r"]), step=1)

        roi_info = {"cx": roi_cx, "cy": roi_cy, "r": roi_r}
        salvar_roi(session_state, escolhido, roi_info)

        mask_roi = criar_mascara_roi(img_bgr.shape, roi_info, barra_info=barra_info)
        prep = preprocessar_para_candidatos(img_bgr, mask_roi)

        img_roi_preview = desenhar_roi_e_circulos(
            img_bgr,
            roi_info,
            obter_circulos(session_state, escolhido),
            titulo="ROI circular"
        )

        st.image(cv_to_pil(img_roi_preview), caption="Pré-visualização da ROI circular", width=760)

        # ---------------- calibração ----------------
        st.markdown("## Calibração")
        if px_per_mm:
            st.success(f"Barra detectada: {px_per_mm:.2f} px para 1,0 mm")
        else:
            st.warning("A barra de 1,0 mm não foi detectada automaticamente.")

        st.image(cv_to_pil(img_calibracao), caption="Detecção da barra de escala", width=420)

        # ---------------- candidatos iniciais ----------------
        st.markdown("## Candidatos iniciais")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Gerar candidatos iniciais", key="btn_gerar_candidatos"):
                candidatos = gerar_candidatos_iniciais(prep["roi_sharpen"], roi_info, px_per_mm)
                salvar_circulos(session_state, escolhido, candidatos)
                st.success(f"{len(candidatos)} candidatos gerados.")
                st.rerun()

        with c2:
            if st.button("Limpar todos os círculos", key="btn_limpar_circulos"):
                salvar_circulos(session_state, escolhido, [])
                st.success("Círculos removidos.")
                st.rerun()

        circles = obter_circulos(session_state, escolhido)

        # ---------------- adicionar manualmente ----------------
        st.markdown("## Adicionar bolha manualmente")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            novo_x = st.number_input("X", min_value=0.0, max_value=float(img_bgr.shape[1]), value=float(roi_info["cx"]), step=1.0, key="novo_x")
        with a2:
            novo_y = st.number_input("Y", min_value=0.0, max_value=float(img_bgr.shape[0]), value=float(roi_info["cy"]), step=1.0, key="novo_y")
        with a3:
            novo_r = st.number_input("Raio", min_value=1.0, max_value=float(min(img_bgr.shape[:2])), value=10.0, step=1.0, key="novo_r")
        with a4:
            st.write("")
            st.write("")
            if st.button("Adicionar círculo", key="btn_add_circle"):
                if ponto_dentro_roi(novo_x, novo_y, roi_info, folga=novo_r + 1):
                    circles.append({"ativo": True, "x": novo_x, "y": novo_y, "r": novo_r})
                    circles = remover_duplicados(circles)
                    salvar_circulos(session_state, escolhido, circles)
                    st.success("Círculo adicionado.")
                    st.rerun()
                else:
                    st.warning("O círculo precisa ficar totalmente dentro da ROI.")

        # ---------------- edição por tabela ----------------
        st.markdown("## Editar / remover círculos")
        df_edit = circles_to_dataframe(circles)

        edited_df = st.data_editor(
            df_edit,
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_circulos_{escolhido}"
        )

        if st.button("Salvar edição dos círculos", key="btn_salvar_edicao"):
            novos = dataframe_to_circles(edited_df, roi_info)
            salvar_circulos(session_state, escolhido, novos)
            st.success("Edição salva.")
            st.rerun()

        circles = obter_circulos(session_state, escolhido)

        # ---------------- imagem final ----------------
        st.markdown("## Resultado final")
        img_resultado = desenhar_roi_e_circulos(
            img_bgr,
            roi_info,
            circles,
            titulo="Bolhas"
        )
        st.image(cv_to_pil(img_resultado), width=900)

        # ---------------- medidas ----------------
        df_med = montar_dataframe_medidas(circles, px_per_mm)

        if df_med.empty:
            st.info("Nenhuma bolha ativa cadastrada.")
            return

        st.markdown("## Tabela automática")
        st.dataframe(df_med, use_container_width=True)

        if "Diâmetro (µm)" in df_med.columns and not df_med["Diâmetro (µm)"].dropna().empty:
            maiores_500 = df_med[df_med["Diâmetro (µm)"] > 500]
            pct_500 = 100.0 * len(maiores_500) / len(df_med)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Bolhas totais", len(df_med))
            s2.metric("Bolhas > 500 µm", len(maiores_500))
            s3.metric("% > 500 µm", f"{pct_500:.2f}%")
            s4.metric("Diâmetro médio (µm)", f"{df_med['Diâmetro (µm)'].mean():.2f}")

            e1, e2 = st.columns(2)
            e1.metric("Mediana (µm)", f"{df_med['Diâmetro (µm)'].median():.2f}")
            e2.metric("Máximo (µm)", f"{df_med['Diâmetro (µm)'].max():.2f}")

            fig_bar, fig_curve, tabela = plotar_distribuicao(df_med)
            if fig_bar is not None:
                st.markdown("## Histograma")
                st.pyplot(fig_bar)

                st.markdown("## Curva")
                st.pyplot(fig_curve)

                st.markdown("## Quantidade por faixa")
                st.dataframe(tabela, use_container_width=True)
