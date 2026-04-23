import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Any


def _safe_upper(texto: Any) -> str:
    if texto is None:
        return ""
    return str(texto).strip().upper()


def extrair_info_tag(tag_teste: str) -> Dict[str, Any]:
    tag = str(tag_teste).strip()
    base = tag.rsplit(".", 1)[0]
    partes = [p for p in base.split("_") if p]

    amostra = None
    teste = None
    tempo_s = None
    codigo_aux = None
    grupo = None

    for p in partes:
        pu = _safe_upper(p)

        if pu.startswith("A") and pu[1:].isdigit():
            amostra = pu
            continue

        if pu.startswith("T") and pu[1:].isdigit():
            teste = pu
            continue

        if pu.endswith("S") and pu[:-1].isdigit():
            tempo_s = int(pu[:-1])
            continue

    if partes:
        ultimo = partes[-1]
        if not _safe_upper(ultimo).endswith("S") and not _safe_upper(ultimo).startswith("T"):
            grupo = ultimo

    for p in partes:
        pu = _safe_upper(p)
        if pu.isdigit():
            codigo_aux = pu
            break

    if grupo is None and len(partes) > 0:
        grupo = partes[-1]

    return {
        "tag_teste": base,
        "amostra": amostra,
        "teste": teste,
        "tempo_s": tempo_s,
        "codigo_aux": codigo_aux,
        "grupo": grupo,
    }


def normalizar_resultados_salvos(resultados: List[Dict[str, Any]]) -> pd.DataFrame:
    linhas = []

    for item in resultados or []:
        tag = item.get("tag_teste", "")
        info = extrair_info_tag(tag)

        tabela_bolhas = item.get("tabela_bolhas", []) or []
        df_bolhas = pd.DataFrame(tabela_bolhas)

        diam_media_um = np.nan
        diam_mediana_um = np.nan
        diam_max_um = np.nan
        diam_min_um = np.nan

        if not df_bolhas.empty and "diametro_um" in df_bolhas.columns:
            serie = pd.to_numeric(df_bolhas["diametro_um"], errors="coerce").dropna()
            if not serie.empty:
                diam_media_um = float(serie.mean())
                diam_mediana_um = float(serie.median())
                diam_max_um = float(serie.max())
                diam_min_um = float(serie.min())

        linhas.append({
            "id": item.get("id"),
            "created_at": item.get("created_at"),
            "tag_teste": info["tag_teste"],
            "amostra": info["amostra"],
            "teste": info["teste"],
            "tempo_s": info["tempo_s"],
            "codigo_aux": info["codigo_aux"],
            "grupo": info["grupo"],
            "percentual_maior_500_um": float(item.get("percentual_bolhas_maiores_500_um", 0.0) or 0.0),
            "quantidade_total_bolhas": int(item.get("quantidade_total_bolhas", 0) or 0),
            "diametro_medio_um": diam_media_um,
            "diametro_mediano_um": diam_mediana_um,
            "diametro_max_um": diam_max_um,
            "diametro_min_um": diam_min_um,
        })

    df = pd.DataFrame(linhas)

    if not df.empty:
        df["grupo"] = df["grupo"].fillna("SEM_GRUPO").astype(str)
        df["tempo_s"] = pd.to_numeric(df["tempo_s"], errors="coerce")

    return df


def montar_resumo_por_grupo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "grupo",
            "qtd_testes",
            "media_percentual_maior_500_um",
            "media_quantidade_total_bolhas",
            "media_diametro_um",
            "mediana_diametro_um",
        ])

    resumo = (
        df.groupby("grupo", dropna=False)
        .agg(
            qtd_testes=("tag_teste", "count"),
            media_percentual_maior_500_um=("percentual_maior_500_um", "mean"),
            media_quantidade_total_bolhas=("quantidade_total_bolhas", "mean"),
            media_diametro_um=("diametro_medio_um", "mean"),
            mediana_diametro_um=("diametro_mediano_um", "mean"),
        )
        .reset_index()
    )

    for col in [
        "media_percentual_maior_500_um",
        "media_quantidade_total_bolhas",
        "media_diametro_um",
        "mediana_diametro_um",
    ]:
        resumo[col] = resumo[col].round(2)

    return resumo


def montar_resumo_por_grupo_tempo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "grupo",
            "tempo_s",
            "qtd_testes",
            "media_percentual_maior_500_um",
            "media_quantidade_total_bolhas",
            "media_diametro_um",
            "mediana_diametro_um",
        ])

    base = df.copy()
    base["tempo_s"] = pd.to_numeric(base["tempo_s"], errors="coerce")

    resumo = (
        base.groupby(["grupo", "tempo_s"], dropna=False)
        .agg(
            qtd_testes=("tag_teste", "count"),
            media_percentual_maior_500_um=("percentual_maior_500_um", "mean"),
            media_quantidade_total_bolhas=("quantidade_total_bolhas", "mean"),
            media_diametro_um=("diametro_medio_um", "mean"),
            mediana_diametro_um=("diametro_mediano_um", "mean"),
        )
        .reset_index()
        .sort_values(["grupo", "tempo_s"], na_position="last")
    )

    for col in [
        "media_percentual_maior_500_um",
        "media_quantidade_total_bolhas",
        "media_diametro_um",
        "mediana_diametro_um",
    ]:
        resumo[col] = resumo[col].round(2)

    return resumo


def dataframe_para_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


def render_resultados_granulometria(listar_resultados_granulometria_supabase):
    st.markdown("### Resultados")

    if st.button("Atualizar resultados", key="btn_atualizar_resultados_supabase"):
        st.rerun()

    resultados, erro = listar_resultados_granulometria_supabase()
    if erro:
        st.error(f"Erro ao carregar resultados do Supabase: {erro}")
        return

    if not resultados:
        st.info("Aqui serão exibidos os resultados das análises.")
        return

    df = normalizar_resultados_salvos(resultados)
    if df.empty:
        st.warning("Existem registros, mas não foi possível montar a tabela de resultados.")
        return

    resumo_grupo = montar_resumo_por_grupo(df)
    resumo_grupo_tempo = montar_resumo_por_grupo_tempo(df)

    st.markdown("### Visão geral")

    total_testes = int(len(df))
    total_grupos = int(df["grupo"].nunique()) if "grupo" in df.columns else 0
    media_global_pct_500 = float(df["percentual_maior_500_um"].mean()) if not df.empty else 0.0
    media_global_qtd = float(df["quantidade_total_bolhas"].mean()) if not df.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total de testes", total_testes)
    with c2:
        st.metric("Total de grupos", total_grupos)
    with c3:
        st.metric("Média % > 500 µm", f"{media_global_pct_500:.2f}%")
    with c4:
        st.metric("Média de bolhas/teste", f"{media_global_qtd:.2f}")

    st.markdown("### Quantidade de testes por grupo")
    df_qtd = resumo_grupo[["grupo", "qtd_testes"]].copy()
    st.bar_chart(df_qtd.set_index("grupo"))
    st.dataframe(df_qtd, use_container_width=True)

    st.markdown("### Médias por grupo")
    st.dataframe(resumo_grupo, use_container_width=True)

    st.markdown("### Médias por grupo e tempo")
    st.dataframe(resumo_grupo_tempo, use_container_width=True)

    df_pct_tempo = resumo_grupo_tempo.dropna(subset=["tempo_s"]).copy()

    if not df_pct_tempo.empty:
        st.markdown("### % de bolhas > 500 µm por tempo")
        pivot_pct = df_pct_tempo.pivot(
            index="tempo_s",
            columns="grupo",
            values="media_percentual_maior_500_um",
        ).sort_index()
        st.line_chart(pivot_pct)

        st.markdown("### Quantidade média de bolhas por tempo")
        pivot_qtd = df_pct_tempo.pivot(
            index="tempo_s",
            columns="grupo",
            values="media_quantidade_total_bolhas",
        ).sort_index()
        st.line_chart(pivot_qtd)

        st.markdown("### Diâmetro médio (µm) por tempo")
        pivot_diam = df_pct_tempo.pivot(
            index="tempo_s",
            columns="grupo",
            values="media_diametro_um",
        ).sort_index()
        st.line_chart(pivot_diam)

    st.markdown("### Tabela bruta dos testes")
    st.dataframe(df, use_container_width=True)

    with st.expander("Diagnóstico — registros crus vindos do Supabase"):
        st.write(resultados)

    st.markdown("### Exportação dos resultados")
    csv_resumo_grupo = dataframe_para_csv_bytes(resumo_grupo)
    csv_resumo_grupo_tempo = dataframe_para_csv_bytes(resumo_grupo_tempo)
    csv_bruto = dataframe_para_csv_bytes(df)

    d1, d2, d3 = st.columns(3)

    with d1:
        st.download_button(
            label="Baixar resumo por grupo (CSV)",
            data=csv_resumo_grupo,
            file_name="resumo_por_grupo.csv",
            mime="text/csv",
            key="download_resumo_grupo",
        )

    with d2:
        st.download_button(
            label="Baixar resumo por grupo e tempo (CSV)",
            data=csv_resumo_grupo_tempo,
            file_name="resumo_por_grupo_tempo.csv",
            mime="text/csv",
            key="download_resumo_grupo_tempo",
        )

    with d3:
        st.download_button(
            label="Baixar tabela bruta (CSV)",
            data=csv_bruto,
            file_name="resultados_brutos_testes.csv",
            mime="text/csv",
            key="download_resultados_brutos",
        )
