import os
import io
import uuid
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import streamlit as st
from supabase import create_client, Client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload

from consulta_imagens import render_consulta_imagens
from resultados_granulometria import render_resultados_granulometria
from supabase_resultados import (
    salvar_resultado_teste_supabase,
    listar_resultados_granulometria_supabase,
)


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="B-Foam MSB",
    page_icon="🔬",
    layout="wide",
)


# ============================================================
# GOOGLE DRIVE
# ============================================================
def get_drive_service():
    creds_dict = dict(st.secrets["gcp_service_account"])

    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )

    return build("drive", "v3", credentials=creds)


def salvar_no_drive(arquivo_bytes, nome_arquivo, mime_type):
    try:
        service = get_drive_service()
        folder_id = st.secrets["google_drive"]["folder_id"]

        file_metadata = {
            "name": nome_arquivo,
            "parents": [folder_id]
        }

        media = MediaIoBaseUpload(
            io.BytesIO(arquivo_bytes),
            mimetype=mime_type,
            resumable=False
        )

        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, name"
        ).execute()

        return file.get("id")

    except Exception as e:
        st.error(f"Erro ao comunicar com o Google Drive: {e}")
        return None


# ============================================================
# SUPABASE
# ============================================================
def get_supabase_config():
    secrets_supabase = st.secrets.get("supabase", {})

    url = (
        st.secrets.get("SUPABASE_URL")
        or secrets_supabase.get("SUPABASE_URL")
        or os.getenv("SUPABASE_URL")
    )

    key = (
        st.secrets.get("SUPABASE_KEY")
        or secrets_supabase.get("SUPABASE_KEY")
        or os.getenv("SUPABASE_KEY")
    )

    bucket = (
        st.secrets.get("SUPABASE_BUCKET")
        or secrets_supabase.get("SUPABASE_BUCKET")
        or os.getenv("SUPABASE_BUCKET")
    )

    return url, key, bucket


@st.cache_resource
def get_supabase_client() -> Client:
    url, key, _ = get_supabase_config()

    if not url or not key:
        raise ValueError("SUPABASE_URL ou SUPABASE_KEY não configurados.")

    return create_client(url, key)


def get_bucket_name() -> str:
    _, _, bucket = get_supabase_config()

    if not bucket:
        raise ValueError("SUPABASE_BUCKET não configurado.")

    return bucket


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

        conteudo = file_obj.getvalue()

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
# AUXILIARES
# ============================================================
def montar_nome_arquivo(amostra, teste, tempo, concentracao, dispositivo, uploaded_file, outro_dispositivo=""):
    c_clean = concentracao.replace(",", "").replace("%", "")
    disp_final = outro_dispositivo if dispositivo == "Outros" else dispositivo

    extensao = uploaded_file.name.split(".")[-1].lower()
    if extensao == "jpg":
        extensao = "jpeg"

    nome_final = (
        f"A{str(amostra).zfill(3)}_"
        f"T{str(teste).zfill(3)}_"
        f"{int(tempo)}s_"
        f"{c_clean}_"
        f"{disp_final}.{extensao}"
    )
    return nome_final, extensao, disp_final


def get_mime_type(uploaded_file, extensao):
    mime_type = uploaded_file.type
    if not mime_type:
        if extensao == "png":
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"
    return mime_type


def carregar_logo_msb_base64() -> str:
    caminhos_possiveis = [
        "logo_msb.png",
        "msb_logo.png",
        "assets/logo_msb.png",
        "assets/msb_logo.png",
        "images/logo_msb.png",
        "images/msb_logo.png",
    ]

    for caminho in caminhos_possiveis:
        p = Path(caminho)
        if p.exists() and p.is_file():
            with open(p, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{data}"

    return ""


def render_banner_bfoam():
    logo_base64 = carregar_logo_msb_base64()

    if logo_base64:
        logo_html = f'<img src="{logo_base64}" class="banner-logo" alt="Logo MSB">'
    else:
        logo_html = '<div class="banner-logo-placeholder">MSB</div>'

    st.markdown(
        f"""
        <div class="banner-topo">
            <div class="banner-logo-area">
                {logo_html}
            </div>
            <div class="banner-texto">
                <div class="banner-titulo">B_Foam</div>
                <div class="banner-subtitulo">Engenharia MSB - Plataforma de Análise</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CSS E ESTILIZAÇÃO
# ============================================================
st.markdown("""
<style>
    .titulo-amarelo {
        color: #FFD700;
        font-size: 2.2em;
        font-weight: bold;
        text-align: center;
        margin: 10px 0 20px 0;
    }

    .subtexto-card {
        color: #B9D1EA;
        font-size: 0.95em;
        margin-top: 8px;
        line-height: 1.35;
        text-align: center;
    }

    .sidebar-link-box {
        border: 1px solid #2E7BCF;
        border-radius: 10px;
        padding: 10px 12px;
        margin-top: 10px;
        background-color: rgba(255,255,255,0.02);
    }

    .sidebar-link-box a {
        color: #B9D1EA !important;
        text-decoration: none;
        font-weight: 600;
    }

    .sidebar-link-box a:hover {
        color: #FFFFFF !important;
        text-decoration: underline;
    }

    .banner-topo {
        width: 100%;
        background: #FFFFFF;
        border-radius: 18px;
        padding: 18px 26px;
        display: flex;
        align-items: center;
        gap: 22px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.12);
        margin: 6px 0 24px 0;
    }

    .banner-logo-area {
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 110px;
    }

    .banner-logo {
        max-height: 78px;
        max-width: 120px;
        object-fit: contain;
    }

    .banner-logo-placeholder {
        width: 88px;
        height: 88px;
        border-radius: 16px;
        background: #E9EEF6;
        color: #0A2A66;
        font-weight: bold;
        font-size: 1.5em;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .banner-texto {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .banner-titulo {
        color: #0A2A66;
        font-size: 2.4em;
        font-weight: 800;
        line-height: 1.0;
        margin-bottom: 6px;
    }

    .banner-subtitulo {
        color: #24456F;
        font-size: 1.05em;
        font-weight: 600;
    }

    .bloco-selecao {
        margin-top: 18px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# NAVEGAÇÃO
# ============================================================
if "pagina" not in st.session_state:
    st.session_state.pagina = "selecao"


def ir_para_cadastro(tipo):
    st.session_state.pagina = "cadastro"
    st.session_state.tipo_selecionado = tipo


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Vídeos de Apoio")
    st.video("https://youtu.be/hY5K55Ha2pg")
    st.write(f"Tutorial: {st.session_state.get('tipo_selecionado', 'Geral')}")

    st.markdown("### Pastas compartilhadas")

    st.markdown(
        """
        <div class="sidebar-link-box">
            📚 <a href="https://drive.google.com/drive/folders/1t0-cqQjqLRbiexGowbBqmmQDoe3RMG2I?usp=sharing" target="_blank">
            Referências
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-link-box">
            📁 <a href="https://drive.google.com/drive/folders/1wfb24h6WLPPMqBnG2FT1jwBbA_bQKGTV?usp=sharing" target="_blank">
            Documentos controlados
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.pagina == "selecao":
        st.divider()
        st.subheader("🔑 Central de Acessos")
        with st.expander("Clique aqui para ver acessos"):
            st.markdown("""
            **E-mail**: `msbbfoam@gmail.com`  
            **Senha**: `Bfoam-50`
            """)


# ============================================================
# TELAS
# ============================================================
if st.session_state.pagina == "selecao":
    render_banner_bfoam()

    st.markdown(
        '<p class="titulo-amarelo">Selecione o tipo de análise:</p>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="bloco-selecao">', unsafe_allow_html=True)

    col_esq, col1, col2, col3, col_dir = st.columns([1, 1.2, 1.2, 1.2, 1])

    with col1:
        if st.button("Teste de Meia-Vida", use_container_width=True):
            ir_para_cadastro("Meia-Vida")
        st.markdown(
            '<div class="subtexto-card">Avalia o tempo de decaimento da espuma.</div>',
            unsafe_allow_html=True
        )

    with col2:
        if st.button("Teste de Granulometria", use_container_width=True):
            ir_para_cadastro("Granulometria")
        st.markdown(
            '<div class="subtexto-card">Mede a distribuição do tamanho das bolhas.</div>',
            unsafe_allow_html=True
        )

    with col3:
        if st.button("Teste de Estabilidade Dinâmica", use_container_width=True):
            ir_para_cadastro("Estabilidade Dinâmica")
        st.markdown(
            '<div class="subtexto-card">Verifica a resistência estrutural da espuma.</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.pagina == "cadastro":
    st.markdown(f"# Ficha de Cadastro: {st.session_state.tipo_selecionado}")

    if st.button("⬅️ Voltar ao Menu Principal"):
        st.session_state.pagina = "selecao"
        st.rerun()

    # ========================================================
    # GRANULOMETRIA
    # ========================================================
    if st.session_state.tipo_selecionado == "Granulometria":
        aba_resultados, aba_cadastro_img, aba_consulta = st.tabs(
            ["📊 Resultados", "➕ Cadastrar Nova Imagem", "🔍 Consultar Imagens"]
        )

        with aba_resultados:
            render_resultados_granulometria(listar_resultados_granulometria_supabase)

        with aba_cadastro_img:
            st.markdown("### Cadastrar Nova Imagem")

            with st.form("form_nova_imagem_granulometria", clear_on_submit=True):
                amostra = st.text_input("Amostra (ex: 001)", key="gran_amostra")
                teste = st.text_input("Teste (ex: 001)", key="gran_teste")
                tempo = st.number_input(
                    "Tempo de estabilidade (segundos)",
                    min_value=0,
                    step=1,
                    key="gran_tempo",
                )

                concentracao = st.selectbox(
                    "Concentração do Polidocanol",
                    ["3,00%", "1,00%", "0,50%", "0,25%"],
                    key="gran_concentracao",
                )

                dispositivo = st.selectbox(
                    "Dispositivo utilizado",
                    ["V08", "V09", "V10", "Tessari", "Outros"],
                    key="gran_dispositivo",
                )

                outro_dispositivo = ""
                if dispositivo == "Outros":
                    outro_dispositivo = st.text_input("Especifique o dispositivo:", key="gran_outro_disp")

                uploaded_file = st.file_uploader(
                    "Escolha a imagem do teste:",
                    type=["png", "jpg", "jpeg"],
                    key="gran_upload_imagem",
                )

                submitted = st.form_submit_button("Salvar Registro no Supabase")

                if submitted:
                    if uploaded_file is None:
                        st.error("Por favor, faça o upload da imagem primeiro.")
                    else:
                        try:
                            nome_final, extensao, _ = montar_nome_arquivo(
                                amostra=amostra,
                                teste=teste,
                                tempo=tempo,
                                concentracao=concentracao,
                                dispositivo=dispositivo,
                                uploaded_file=uploaded_file,
                                outro_dispositivo=outro_dispositivo,
                            )

                            mime_type = get_mime_type(uploaded_file, extensao)

                            with st.spinner("Salvando no Supabase..."):
                                path_salvo, erro = upload_imagem_supabase(
                                    uploaded_file,
                                    nome_destino=nome_final,
                                )

                            if erro:
                                st.error(f"Erro ao salvar no Supabase: {erro}")
                            else:
                                st.success(f"Arquivo salvo com sucesso! Nome: {path_salvo}")

                        except Exception as e:
                            st.error(f"Erro no cadastro da imagem: {e}")

        with aba_consulta:
            render_consulta_imagens(
                listar_imagens_supabase=listar_imagens_supabase,
                montar_url_publica=montar_url_publica,
                session_state=st.session_state,
                salvar_resultado_teste=salvar_resultado_teste_supabase,
            )

    # ========================================================
    # OUTRAS ANÁLISES (mantém comportamento antigo no Drive)
    # ========================================================
    else:
        with st.form("form_novo_teste", clear_on_submit=True):
            amostra = st.text_input("Amostra (ex: 001)")
            teste = st.text_input("Teste (ex: 001)")
            tempo = st.number_input(
                "Tempo de estabilidade (segundos)",
                min_value=0,
                step=1
            )

            concentracao = st.selectbox(
                "Concentração do Polidocanol",
                ["3,00%", "1,00%", "0,50%", "0,25%"]
            )

            dispositivo = st.selectbox(
                "Dispositivo utilizado",
                ["V08", "V09", "V10", "Tessari", "Outros"]
            )

            outro_dispositivo = ""
            if dispositivo == "Outros":
                outro_dispositivo = st.text_input("Especifique o dispositivo:")

            uploaded_file = st.file_uploader(
                "Escolha a imagem do teste:",
                type=["png", "jpg", "jpeg"]
            )

            submitted = st.form_submit_button("Salvar Registro no Drive")

            if submitted:
                if uploaded_file is None:
                    st.error("Por favor, faça o upload da imagem primeiro.")
                else:
                    nome_final, extensao, _ = montar_nome_arquivo(
                        amostra=amostra,
                        teste=teste,
                        tempo=tempo,
                        concentracao=concentracao,
                        dispositivo=dispositivo,
                        uploaded_file=uploaded_file,
                        outro_dispositivo=outro_dispositivo,
                    )

                    mime_type = get_mime_type(uploaded_file, extensao)

                    with st.spinner("Salvando no Drive..."):
                        file_id = salvar_no_drive(
                            uploaded_file.getvalue(),
                            nome_final,
                            mime_type
                        )

                    if file_id:
                        st.success(f"Arquivo salvo com sucesso! Nome: {nome_final}")
                    else:
                        st.error("O arquivo não foi salvo no Drive.")
