import requests
import streamlit as st


def get_image_as_base64(path):
    try:
        with open(path, "rb") as image_file:
            import base64
            data = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return ""
        
# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="B-Foam MSB",
    page_icon="🔬",
    layout="centered"
)

# ============================================================
# CONFIGURAÇÕES
# ============================================================
BUCKET_NAME = "imbfoam"


# ============================================================
# FUNÇÕES SUPABASE
# ============================================================
def get_supabase_base_url():
    return st.secrets["supabase"]["SUPABASE_URL"].strip().rstrip("/")


def get_supabase_key():
    return st.secrets["supabase"]["SUPABASE_KEY"].strip()


def salvar_no_supabase(uploaded_file, nome_arquivo, mime_type):
    try:
        base_url = get_supabase_base_url()
        api_key = get_supabase_key()

        url = f"{base_url}/storage/v1/object/{BUCKET_NAME}/{nome_arquivo}"

        headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": mime_type,
            "x-upsert": "true"
        }

        response = requests.post(
            url,
            headers=headers,
            data=uploaded_file.getvalue(),
            timeout=60
        )

        if response.ok:
            return True, response.text

        return False, f"{response.status_code} - {response.text}"

    except Exception as e:
        return False, repr(e)


def listar_imagens_supabase(search_text=""):
    try:
        base_url = get_supabase_base_url()
        api_key = get_supabase_key()

        url = f"{base_url}/storage/v1/object/list/{BUCKET_NAME}"

        headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "prefix": "",
            "limit": 100,
            "offset": 0,
            "sortBy": {"column": "name", "order": "desc"}
        }

        if search_text.strip():
            payload["search"] = search_text.strip()

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60
        )

        if not response.ok:
            return [], f"{response.status_code} - {response.text}"

        dados = response.json()

        imagens = []
        for item in dados:
            nome = item.get("name", "")
            if nome.lower().endswith((".png", ".jpg", ".jpeg")):
                imagens.append(item)

        return imagens, None

    except Exception as e:
        return [], repr(e)


def montar_url_publica(nome_arquivo):
    base_url = get_supabase_base_url()
    return f"{base_url}/storage/v1/object/public/{BUCKET_NAME}/{nome_arquivo}"


# ============================================================
# CSS E ESTILIZAÇÃO
# ============================================================
st.markdown("""
<style>
    .titulo-amarelo {
        color: #FFD700;
        font-size: 3.2em;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }

    .bloco-principal {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 20px;
        margin-top: 10px;
    }

    .subtitulo-secao {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .topo-card {
        background-color: white;
        border-radius: 12px;
        padding: 18px 24px;
        margin: 10px auto 24px auto;
        max-width: 640px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 22px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.18);
    }

    .topo-logo {
        height: 56px;
        width: auto;
    }

    .topo-titulo {
        color: black;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
    }

    .topo-subtitulo {
        color: #444;
        font-size: 0.95rem;
        margin: 4px 0 0 0;
    }

    .subtitulo-home {
        color: #FFD700;
        font-size: 0.95rem;
        font-weight: 700;
        text-align: center;
        margin: 10px 0 14px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ESTADO
# ============================================================
if "pagina" not in st.session_state:
    st.session_state.pagina = "selecao"

if "tipo_selecionado" not in st.session_state:
    st.session_state.tipo_selecionado = ""

if "modo_cadastro" not in st.session_state:
    st.session_state.modo_cadastro = "Cadastrar nova imagem"

if "lista_imagens_consulta" not in st.session_state:
    st.session_state.lista_imagens_consulta = []

if "ultimo_termo_busca" not in st.session_state:
    st.session_state.ultimo_termo_busca = None


# ============================================================
# FUNÇÕES DE NAVEGAÇÃO
# ============================================================
def ir_para_cadastro(tipo):
    st.session_state.pagina = "cadastro"
    st.session_state.tipo_selecionado = tipo
    st.session_state.modo_cadastro = "Cadastrar nova imagem"


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Vídeos de Apoio")
    st.video("https://youtu.be/hY5K55Ha2pg")
    st.write(f"Tutorial: {st.session_state.get('tipo_selecionado', 'Geral')}")


# ============================================================
# TELA PRINCIPAL
# ============================================================
if st.session_state.pagina == "selecao":
    logo_base64 = get_image_as_base64("logo-msb.png")

    st.markdown(f"""
    <div class="topo-card">
        <img src="{logo_base64}" class="topo-logo">
        <div>
            <p class="topo-titulo">B-Foam</p>
            <p class="topo-subtitulo">Engenharia MSB · Plataforma de Análise</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="subtitulo-home">Selecione o tipo de análise desejada:</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Teste de Meia-Vida", use_container_width=True):
        ir_para_cadastro("Meia-Vida")
    st.caption("Avalia o tempo de decaimento da espuma.")

with c2:
    if st.button("Teste de Granulometria", use_container_width=True):
        ir_para_cadastro("Granulometria")
    st.caption("Mede a distribuição do tamanho das bolhas.")

with c3:
    if st.button("Teste de Estabilidade Dinâmica", use_container_width=True):
        ir_para_cadastro("Estabilidade Dinâmica")
    st.caption("Verifica a resistência estrutural da espuma.")


# ============================================================
# TELA DE CADASTRO / CONSULTA
# ============================================================
elif st.session_state.pagina == "cadastro":
    st.title(f"Ficha de Cadastro: {st.session_state.tipo_selecionado}")

    if st.button("⬅️ Voltar ao Menu Principal"):
        st.session_state.pagina = "selecao"
        st.rerun()

    st.markdown("## Ação")

    modo = st.radio(
        "Escolha a ação",
        ["Cadastrar nova imagem", "Consultar imagem"],
        horizontal=True,
        key="modo_cadastro"
    )

    # ========================================================
    # MODO: CADASTRAR NOVA IMAGEM
    # ========================================================
    if modo == "Cadastrar nova imagem":
        with st.container(border=True):
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

                submitted = st.form_submit_button("Salvar Registro no Supabase")

                if submitted:
                    if uploaded_file is None:
                        st.error("Por favor, faça o upload da imagem primeiro.")
                    else:
                        c_clean = concentracao.replace(",", "").replace("%", "")
                        disp_final = outro_dispositivo.strip() if dispositivo == "Outros" else dispositivo

                        if dispositivo == "Outros" and not disp_final:
                            st.error("Por favor, informe o nome do dispositivo.")
                        else:
                            extensao = uploaded_file.name.split(".")[-1].lower()
                            if extensao == "jpg":
                                extensao = "jpeg"

                            nome_final = (
                                f"A{amostra.zfill(3)}_"
                                f"T{teste.zfill(3)}_"
                                f"{tempo}s_"
                                f"{c_clean}_"
                                f"{disp_final}.{extensao}"
                            )

                            mime_type = uploaded_file.type
                            if not mime_type:
                                mime_type = "image/png" if extensao == "png" else "image/jpeg"

                            with st.spinner("Enviando para o Supabase..."):
                                sucesso, detalhe = salvar_no_supabase(
                                    uploaded_file,
                                    nome_final,
                                    mime_type
                                )

                            if sucesso:
                                st.success(f"Arquivo salvo com sucesso: {nome_final}")
                                st.info(f"URL pública: {montar_url_publica(nome_final)}")

                                # Atualiza a lista para consulta imediata
                                st.session_state.lista_imagens_consulta = []
                                st.session_state.ultimo_termo_busca = None
                            else:
                                st.error(f"Erro ao salvar no Supabase: {detalhe}")

    # ========================================================
    # MODO: CONSULTAR IMAGEM
    # ========================================================
    elif modo == "Consultar imagem":
        with st.container(border=True):
            st.markdown("## Consultar imagem salva")

            termo_busca = st.text_input(
                "Buscar por nome do arquivo",
                placeholder="Ex.: A000, V08, 300, png..."
            )

            col1, col2 = st.columns([1, 3])
            with col1:
                atualizar = st.button("Atualizar lista", use_container_width=True)

            precisa_atualizar = (
                atualizar
                or st.session_state.ultimo_termo_busca != termo_busca
                or not st.session_state.lista_imagens_consulta
            )

            if precisa_atualizar:
                with st.spinner("Consultando imagens no Supabase..."):
                    imagens, erro = listar_imagens_supabase(termo_busca)

                st.session_state.ultimo_termo_busca = termo_busca

                if erro:
                    st.session_state.lista_imagens_consulta = []
                    st.error(f"Erro ao listar imagens: {erro}")
                else:
                    st.session_state.lista_imagens_consulta = [img["name"] for img in imagens]

            lista_imagens = st.session_state.get("lista_imagens_consulta", [])

            if not lista_imagens:
                st.warning("Nenhuma imagem encontrada.")
            else:
                arquivo_escolhido = st.selectbox(
                    "Selecione a imagem",
                    lista_imagens,
                    index=0
                )

                if arquivo_escolhido:
                    url_publica = montar_url_publica(arquivo_escolhido)

                    st.markdown("### Pré-visualização")
                    st.image(
                        url_publica,
                        caption=arquivo_escolhido,
                        use_container_width=True
                    )

                    st.write(f"**Arquivo:** {arquivo_escolhido}")
                    st.write(f"**URL pública:** {url_publica}")
