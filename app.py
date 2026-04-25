import os
import io
import uuid
import base64
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from supabase import create_client, Client
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload
from manutencao import verificar_manutencao
from manufatura import render_manufatura
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

verificar_manutencao()
LOGO_PATH = "logo-msb.png"


# ============================================================
# CSS GLOBAL
# ============================================================
st.markdown("""
<style>
    /* ============================================================
       COMPACTAÇÃO GLOBAL DO APP
       ============================================================ */

    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 0.8rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
        max-width: 100% !important;
    }

    h1 {
        font-size: 2.0rem !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.6rem !important;
        line-height: 1.12 !important;
    }

    h2 {
        font-size: 1.45rem !important;
        margin-top: 0.7rem !important;
        margin-bottom: 0.45rem !important;
        line-height: 1.15 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        margin-top: 0.6rem !important;
        margin-bottom: 0.35rem !important;
        line-height: 1.15 !important;
    }

    p, label, span {
        font-size: 0.88rem !important;
    }

    .stMarkdown {
        margin-bottom: 0.25rem !important;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.35rem !important;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.6rem !important;
    }

    .stButton > button {
        padding: 0.28rem 0.65rem !important;
        font-size: 0.85rem !important;
        min-height: 32px !important;
        border-radius: 8px !important;
    }

    .stTextInput input,
    .stNumberInput input {
        font-size: 0.85rem !important;
        min-height: 34px !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        font-size: 0.85rem !important;
        min-height: 34px !important;
    }

    .stFileUploader {
        font-size: 0.85rem !important;
    }

    [data-testid="stImage"] img {
        max-height: 460px !important;
        object-fit: contain !important;
    }

    [data-baseweb="tab"] {
        font-size: 0.85rem !important;
        padding: 0.3rem 0.6rem !important;
    }

    [data-baseweb="tab-list"] {
        gap: 0.2rem !important;
    }

    .stAlert {
        padding-top: 0.45rem !important;
        padding-bottom: 0.45rem !important;
        font-size: 0.88rem !important;
    }

    hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* ============================================================
       SIDEBAR
       ============================================================ */

    section[data-testid="stSidebar"] {
        width: 260px !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 1.05rem !important;
        margin-bottom: 0.4rem !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        font-size: 0.84rem !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        font-size: 0.85rem !important;
        min-height: 34px !important;
    }

    .sidebar-link-box {
        border: 1px solid #2E7BCF;
        border-radius: 9px;
        padding: 8px 10px;
        margin-top: 7px;
        background-color: rgba(255,255,255,0.02);
    }

    .sidebar-link-box a {
        color: #B9D1EA !important;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.86rem !important;
    }

    .sidebar-link-box a:hover {
        color: #FFFFFF !important;
        text-decoration: underline;
    }

    /* ============================================================
       BANNER SUPERIOR COM LOGO
       ============================================================ */

    .banner-topo {
    width: 100%;
    background: #FFFFFF;
    border-radius: 0 0 18px 18px;
    padding: 24px 28px !important;
    min-height: 140px;
    display: flex;
    align-items: center;
    gap: 24px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    margin: 0 0 18px 0 !important;
    overflow: visible !important;
}

    .banner-logo-area {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 180px;
    height: 105px;
    overflow: visible !important;
}

.banner-logo {
    max-height: 105px !important;
    max-width: 180px !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    display: block;
}
    .banner-texto {
        display: flex;
        flex-direction: column;
        justify-content: center;
        overflow: visible !important;
    }

    .banner-titulo {
        color: #0A2A66;
        font-size: 1.85em !important;
        font-weight: 800;
        line-height: 1.0;
        margin-bottom: 5px;
    }

    .banner-subtitulo {
        color: #24456F;
        font-size: 0.9em !important;
        font-weight: 600;
    }

    /* ============================================================
       TELA INICIAL
       ============================================================ */

    .titulo-amarelo {
        color: #FFD700;
        font-size: 1.55em !important;
        font-weight: bold;
        text-align: center;
        margin: 6px 0 12px 0 !important;
    }

    .subtexto-card {
        color: #B9D1EA;
        font-size: 0.82em !important;
        margin-top: 5px;
        line-height: 1.2;
        text-align: center;
    }

    .bloco-selecao {
        margin-top: 10px;
    }

    /* ============================================================
       LOGIN
       ============================================================ */

    .login-wrapper {
        width: 100%;
        padding-top: 10px;
    }

    .login-box {
        width: 100%;
        max-width: 480px;
        margin: 0 auto 12px auto;
        background: #FFFFFF;
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.12);
        text-align: center;
    }

    .login-title {
        color: #0A2A66;
        font-size: 1.55em !important;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .login-subtitle {
        color: #35557C;
        font-size: 0.9em !important;
        font-weight: 500;
        margin-bottom: 3px;
    }

    .login-form-box {
        width: 100%;
        max-width: 620px;
        margin: 0 auto;
        padding: 12px;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        background: rgba(255,255,255,0.02);
    }

    /* ============================================================
       FORMULÁRIOS MAIS COMPACTOS
       ============================================================ */

    .compact-form-box {
        max-width: 950px;
        padding: 12px 16px;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 10px;
        background: rgba(255,255,255,0.015);
        margin-top: 8px;
    }

    .compact-title {
        font-size: 1.05rem !important;
        font-weight: 700;
        margin-top: 0.5rem !important;
        margin-bottom: 0.4rem !important;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONTROLE DE ACESSO POR SENHA
# ============================================================
def get_app_password():
    return (
        st.secrets.get("APP_PASSWORD")
        or st.secrets.get("app", {}).get("APP_PASSWORD")
        or os.getenv("APP_PASSWORD")
    )


def carregar_logo_msb_base64() -> str:
    caminhos_possiveis = [
        LOGO_PATH,
        "logo-msb.png",
        "logo_msb.png",
    ]

    for caminho in caminhos_possiveis:
        p = Path(caminho)

        if p.exists() and p.is_file():
            sufixo = p.suffix.lower()
            mime = "image/png"

            if sufixo in [".jpg", ".jpeg"]:
                mime = "image/jpeg"

            with open(p, "rb") as f:
                data = base64.b64encode(f.read()).decode()

            return f"data:{mime};base64,{data}"

    return ""


def render_banner_bfoam():
    logo_base64 = carregar_logo_msb_base64()

    if logo_base64:
        logo_html = f'<img src="{logo_base64}" class="banner-logo" alt="Logo MSB">'
    else:
        logo_html = '<div style="color:#0A2A66;font-weight:700;">Logo não encontrada</div>'

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


def render_login_page():
    render_banner_bfoam()

    st.markdown('<div class="login-wrapper">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="login-box">
            <div class="login-title">Acesso restrito</div>
            <div class="login-subtitle">Digite a senha para acessar a plataforma.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="login-form-box">', unsafe_allow_html=True)

    with st.form("form_login_app", clear_on_submit=False):
        senha = st.text_input("Senha", type="password")
        entrar = st.form_submit_button("Entrar")

        if entrar:
            senha_correta = get_app_password()

            if not senha_correta:
                st.error("APP_PASSWORD não configurada nos secrets.")
            elif senha == senha_correta:
                st.session_state.app_autenticado = True
                st.rerun()
            else:
                st.error("Senha incorreta.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def garantir_login_total():
    if "app_autenticado" not in st.session_state:
        st.session_state.app_autenticado = False

    if not st.session_state.app_autenticado:
        render_login_page()
        st.stop()


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

        resposta = supabase.storage.from_(bucket).list(
            path=prefixo if prefixo else ""
        )

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


def upload_imagem_supabase(
    file_obj,
    nome_destino: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
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
def montar_nome_arquivo(
    amostra,
    teste,
    tempo,
    concentracao,
    dispositivo,
    uploaded_file,
    outro_dispositivo=""
):
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


# ============================================================
# NAVEGAÇÃO
# ============================================================
if "pagina" not in st.session_state:
    st.session_state.pagina = "selecao"

if "confirmou_cadastro_imagem" not in st.session_state:
    st.session_state.confirmou_cadastro_imagem = False


def ir_para_cadastro(tipo):
    st.session_state.pagina = "cadastro"
    st.session_state.tipo_selecionado = tipo

    if tipo != "Granulometria":
        st.session_state.confirmou_cadastro_imagem = False


# ============================================================
# LOGIN TOTAL ANTES DE TUDO
# ============================================================
garantir_login_total()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Vídeos de Apoio")
    st.video("https://youtu.be/KkjUeiGazEc")
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

    st.divider()

    st.markdown("### Links de apoio")

    if st.button("🏭 Processo de fabricação", use_container_width=True):
        st.session_state.pagina = "manufatura"
        st.session_state.confirmou_cadastro_imagem = False
        st.rerun()

    if st.session_state.pagina == "selecao":
        st.divider()
        st.subheader("🔑 Central de Acessos")

        with st.expander("Clique aqui para ver acessos"):
            st.markdown("""
            **E-mail**: `msbbfoam@gmail.com`  
            **Senha**: `Bfoam-50`
            """)

    st.divider()

    if st.button("Encerrar sessão"):
        st.session_state.app_autenticado = False
        st.session_state.confirmou_cadastro_imagem = False
        st.rerun()


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

    col_esq, col1, col2, col3, col_dir = st.columns([1, 1.1, 1.1, 1.1, 1])

    with col1:
        if st.button("Teste de Meia-Vida", use_container_width=True):
            ir_para_cadastro("Meia-Vida")
            st.rerun()

        st.markdown(
            '<div class="subtexto-card">Avalia o tempo de decaimento da espuma.</div>',
            unsafe_allow_html=True
        )

    with col2:
        if st.button("Teste de Granulometria", use_container_width=True):
            ir_para_cadastro("Granulometria")
            st.rerun()

        st.markdown(
            '<div class="subtexto-card">Mede a distribuição do tamanho das bolhas.</div>',
            unsafe_allow_html=True
        )

    with col3:
        if st.button("Teste de Estabilidade Dinâmica", use_container_width=True):
            ir_para_cadastro("Estabilidade Dinâmica")
            st.rerun()

        st.markdown(
            '<div class="subtexto-card">Verifica a resistência estrutural da espuma.</div>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.pagina == "manufatura":
    render_banner_bfoam()

    if st.button("⬅️ Voltar ao Menu Principal"):
        st.session_state.pagina = "selecao"
        st.rerun()

    render_manufatura()


elif st.session_state.pagina == "cadastro":
    render_banner_bfoam()

    st.markdown(f"# Ficha de Cadastro: {st.session_state.tipo_selecionado}")

    if st.button("⬅️ Voltar ao Menu Principal"):
        st.session_state.pagina = "selecao"
        st.session_state.confirmou_cadastro_imagem = False
        st.rerun()

    if st.session_state.tipo_selecionado == "Granulometria":
        aba_resultados, aba_consulta, aba_cadastro_img = st.tabs(
            ["📊 Resultados", "🔍 Consultar Imagens", "➕ Cadastrar Nova Imagem"]
        )

        # ============================================================
        # ABA RESULTADOS
        # ============================================================
        with aba_resultados:
            render_resultados_granulometria(
                listar_resultados_granulometria_supabase
            )

        # ============================================================
        # ABA CONSULTAR IMAGENS
        # ============================================================
        with aba_consulta:
            render_consulta_imagens(
                listar_imagens_supabase=listar_imagens_supabase,
                montar_url_publica=montar_url_publica,
                session_state=st.session_state,
                salvar_resultado_teste=salvar_resultado_teste_supabase,
            )

        # ============================================================
        # ABA CADASTRAR NOVA IMAGEM
        # ============================================================
        with aba_cadastro_img:
            st.markdown('<p class="compact-title">Cadastrar Nova Imagem</p>', unsafe_allow_html=True)

            if not st.session_state.confirmou_cadastro_imagem:
                st.warning(
                    "⚠️ Área exclusiva para testes de imagens. "
                    "O cadastro de uma nova imagem irá adicionar um novo registro ao sistema."
                )

                st.info("Você realmente deseja cadastrar uma nova imagem?")

                col_aviso1, col_aviso2, col_vazio = st.columns([1, 1, 2])

                with col_aviso1:
                    if st.button("✅ Sim, cadastrar", use_container_width=True):
                        st.session_state.confirmou_cadastro_imagem = True
                        st.rerun()

                with col_aviso2:
                    if st.button("⬅️ Não, voltar", use_container_width=True):
                        st.session_state.pagina = "selecao"
                        st.session_state.confirmou_cadastro_imagem = False
                        st.rerun()

            else:
                st.success("Acesso liberado para cadastro de imagem.")

                col_cancelar, col_vazio = st.columns([1, 3])

                with col_cancelar:
                    if st.button("🔒 Bloquear cadastro", use_container_width=True):
                        st.session_state.confirmou_cadastro_imagem = False
                        st.rerun()

                st.markdown('<div class="compact-form-box">', unsafe_allow_html=True)

                with st.form("form_nova_imagem_granulometria", clear_on_submit=True):
                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col1:
                        amostra = st.text_input("Amostra", placeholder="Ex: 001", key="gran_amostra")

                    with col2:
                        teste = st.text_input("Teste", placeholder="Ex: 001", key="gran_teste")

                    with col3:
                        tempo = st.number_input(
                            "Tempo de estabilidade (s)",
                            min_value=0,
                            step=1,
                            key="gran_tempo",
                        )

                    col4, col5, col6 = st.columns([1, 1, 1])

                    with col4:
                        concentracao = st.selectbox(
                            "Concentração",
                            ["3,00%", "1,00%", "0,50%", "0,25%"],
                            key="gran_concentracao",
                        )

                    with col5:
                        dispositivo = st.selectbox(
                            "Dispositivo",
                            ["V08", "V09", "V10", "Tessari", "Outros"],
                            key="gran_dispositivo",
                        )

                    outro_dispositivo = ""

                    with col6:
                        if dispositivo == "Outros":
                            outro_dispositivo = st.text_input(
                                "Outro dispositivo",
                                key="gran_outro_disp"
                            )
                        else:
                            st.text_input(
                                "Outro dispositivo",
                                value="",
                                disabled=True,
                                key="gran_outro_disp_disabled"
                            )

                    uploaded_file = st.file_uploader(
                        "Imagem do teste",
                        type=["png", "jpg", "jpeg"],
                        key="gran_upload_imagem",
                    )

                    col_salvar, col_vazio2 = st.columns([1, 3])

                    with col_salvar:
                        submitted = st.form_submit_button("Salvar no Supabase", use_container_width=True)

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

                                with st.spinner("Salvando no Supabase..."):
                                    path_salvo, erro = upload_imagem_supabase(
                                        uploaded_file,
                                        nome_destino=nome_final,
                                    )

                                if erro:
                                    st.error(f"Erro ao salvar no Supabase: {erro}")
                                else:
                                    st.success(
                                        f"Arquivo salvo com sucesso! Nome: {path_salvo}"
                                    )

                            except Exception as e:
                                st.error(f"Erro no cadastro da imagem: {e}")

                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="compact-form-box">', unsafe_allow_html=True)

        with st.form("form_novo_teste", clear_on_submit=True):
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                amostra = st.text_input("Amostra", placeholder="Ex: 001")

            with col2:
                teste = st.text_input("Teste", placeholder="Ex: 001")

            with col3:
                tempo = st.number_input(
                    "Tempo de estabilidade (s)",
                    min_value=0,
                    step=1
                )

            col4, col5, col6 = st.columns([1, 1, 1])

            with col4:
                concentracao = st.selectbox(
                    "Concentração",
                    ["3,00%", "1,00%", "0,50%", "0,25%"]
                )

            with col5:
                dispositivo = st.selectbox(
                    "Dispositivo",
                    ["V08", "V09", "V10", "Tessari", "Outros"]
                )

            outro_dispositivo = ""

            with col6:
                if dispositivo == "Outros":
                    outro_dispositivo = st.text_input("Outro dispositivo")
                else:
                    st.text_input("Outro dispositivo", value="", disabled=True)

            uploaded_file = st.file_uploader(
                "Imagem do teste",
                type=["png", "jpg", "jpeg"]
            )

            col_salvar, col_vazio = st.columns([1, 3])

            with col_salvar:
                submitted = st.form_submit_button("Salvar no Drive", use_container_width=True)

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

        st.markdown('</div>', unsafe_allow_html=True)
