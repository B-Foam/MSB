import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import streamlit as st
from supabase import create_client, Client

from consulta_imagens import render_consulta_imagens
from resultados_granulometria import render_resultados_granulometria
from supabase_resultados import (
    salvar_resultado_teste_supabase,
    listar_resultados_granulometria_supabase,
)


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Ficha de Cadastro: Granulometria",
    layout="wide",
)

PRIMARY_COLOR = "#0A2A66"


# ============================================================
# SUPABASE
# ============================================================
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
    key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

    if not url or not key:
        raise ValueError("SUPABASE_URL ou SUPABASE_KEY não configurados.")

    return create_client(url, key)


def get_bucket_name() -> str:
    bucket = st.secrets.get("SUPABASE_BUCKET", os.getenv("SUPABASE_BUCKET"))
    if not bucket:
        raise ValueError("SUPABASE_BUCKET não configurado.")
    return bucket


# ============================================================
# FUNÇÕES DE STORAGE
# ============================================================
def listar_imagens_supabase(prefixo: str = "") -> Tuple[List[Dict], Optional[str]]:
    try:
        supabase = get_supabase_client()
        bucket = get_bucket_name()

        resposta = supabase.storage.from_(bucket).list(path=prefixo if prefixo else "")

        if resposta is None:
            return [], None

        imagens = []
        for item in resposta:
            nome = item.get("name")
            if not nome:
                continue

            nome_lower = nome.lower()
            if nome_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                path_completo = f"{prefixo}/{nome}" if prefixo else nome
                imagens.append(
                    {
                        "name": nome,
                        "path": path_completo,
                        "metadata": item,
                    }
                )

        imagens = sorted(imagens, key=lambda x: x["name"].lower())
        return imagens, None

    except Exception as e:
        return [], str(e)


def montar_url_publica(caminho_arquivo: str) -> str:
    supabase = get_supabase_client()
    bucket = get_bucket_name()
    return supabase.storage.from_(bucket).get_public_url(caminho_arquivo)


def upload_imagem_supabase(file_obj, nome_destino: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    try:
        supabase = get_supabase_client()
        bucket = get_bucket_name()

        nome_original = file_obj.name
        extensao = os.path.splitext(nome_original)[1].lower()

        if nome_destino is None or not nome_destino.strip():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_destino = f"{timestamp}_{uuid.uuid4().hex[:8]}{extensao}"
        else:
            if not nome_destino.lower().endswith(extensao):
                nome_destino = f"{nome_destino}{extensao}"

        conteudo = file_obj.read()

        supabase.storage.from_(bucket).upload(
            path=nome_destino,
            file=conteudo,
            file_options={
                "content-type": file_obj.type or "application/octet-stream",
                "upsert": "false",
            },
        )

        return nome_destino, None

    except Exception as e:
        return None, str(e)


# ============================================================
# UI HELPERS
# ============================================================
def render_header():
    st.markdown("# Ficha de Cadastro: Granulometria")

    if st.button("⬅️ Voltar ao Menu Principal", key="btn_voltar_menu"):
        st.session_state["aba_principal"] = "Resultados"
        st.rerun()


def render_aba_cadastro():
    st.markdown("### Cadastrar Nova Imagem")

    with st.container(border=True):
        arquivos = st.file_uploader(
            "Selecione uma ou mais imagens",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
            accept_multiple_files=True,
            key="upload_imagens_granulometria",
        )

        nome_personalizado = st.text_input(
            "Prefixo opcional para o nome do arquivo",
            value="",
            key="prefixo_nome_upload",
            help="Ex.: A001_T001_120s_300_V08",
        )

        if st.button("Enviar imagem(ns)", key="btn_enviar_imagens"):
            if not arquivos:
                st.warning("Selecione ao menos uma imagem.")
                return

            progresso = st.progress(0)
            total = len(arquivos)
            enviados = 0
            erros = []

            for i, arq in enumerate(arquivos, start=1):
                prefixo = nome_personalizado.strip()
                nome_destino = None

                if prefixo:
                    base, ext = os.path.splitext(arq.name)
                    nome_destino = f"{prefixo}_{base}"

                path_salvo, erro = upload_imagem_supabase(arq, nome_destino=nome_destino)
                if erro:
                    erros.append(f"{arq.name}: {erro}")
                else:
                    enviados += 1

                progresso.progress(i / total)

            if enviados > 0:
                st.success(f"{enviados} imagem(ns) enviada(s) com sucesso.")
            if erros:
                st.error("Ocorreram erros no upload:")
                for err in erros:
                    st.write(f"- {err}")


def render_aba_resultados():
    render_resultados_granulometria(listar_resultados_granulometria_supabase)


def render_aba_consulta():
    render_consulta_imagens(
        listar_imagens_supabase=listar_imagens_supabase,
        montar_url_publica=montar_url_publica,
        session_state=st.session_state,
        salvar_resultado_teste=salvar_resultado_teste_supabase,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    render_header()

    aba_resultados, aba_cadastro, aba_consulta = st.tabs(
        ["📊 Resultados", "➕ Cadastrar Nova Imagem", "🔍 Consultar Imagens"]
    )

    with aba_resultados:
        render_aba_resultados()

    with aba_cadastro:
        render_aba_cadastro()

    with aba_consulta:
        render_aba_consulta()


if __name__ == "__main__":
    main()
