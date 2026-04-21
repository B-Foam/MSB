import requests
import streamlit as st
from supabase import create_client

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="B-Foam MSB",
    page_icon="🔬",
    layout="centered"
)

# --- CONFIGURAÇÃO DO SUPABASE ---
BUCKET_NAME = "imbfoam"


def get_supabase_client():
    url = st.secrets["supabase"]["SUPABASE_URL"].strip().rstrip("/")
    key = st.secrets["supabase"]["SUPABASE_KEY"].strip()
    return create_client(url, key)


def get_supabase_base_url():
    return st.secrets["supabase"]["SUPABASE_URL"].strip().rstrip("/")


def salvar_no_supabase(uploaded_file, nome_arquivo, mime_type):
    try:
        base_url = get_supabase_base_url()
        api_key = st.secrets["supabase"]["SUPABASE_KEY"].strip()

        url = f"{base_url}/storage/v1/object/{BUCKET_NAME}/{nome_arquivo}"

        headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": mime_type,
            "x-upsert": "true",
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
    """
    Lista arquivos do bucket usando o SDK do Supabase.
    Requer permissão SELECT em storage.objects.
    """
    try:
        supabase = get_supabase_client()

        options = {
            "limit": 100,
            "offset": 0,
            "sortBy": {"column": "name", "order": "desc"},
        }

        if search_text.strip():
            options["search"] = search_text.strip()

        response = (
            supabase.storage
            .from_(BUCKET_NAME)
            .list("", options)
        )

        arquivos = response if isinstance(response, list) else []

        # Mantém só imagens
        imagens = []
        for item in arquivos:
            nome = item.get("name", "")
            nome_lower = nome.lower()
            if nome_lower.endswith((".png", ".jpg", ".jpeg")):
                imagens.append(item)

        return imagens, None

    except Exception as e:
        return [], repr(e)


def montar_url_publica(nome_arquivo):
    base_url = get_supabase_base_url()
    return f"{base_url}/storage/v1/object/public/{BUCKET_NAME}/{nome_arquivo}"


# --- CSS E ESTILIZAÇÃO ---
st.markdown("""
<style>
    .titulo-amarelo {
        color: #FFD700;
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- LÓGICA DE NAVEGAÇÃO ---
if "pagina" not in st.session_state:
    st.session_state.pagina = "selecao"

if "tipo_selecionado" not in st.session_state:
    st.session_state.tipo_selecionado = ""

if "modo_cadastro" not in st.session_state:
    st.session_state.modo_cadastro = "Cadastrar nova imagem"


def ir_para_cadastro(tipo):
    st.session_state.pagina = "cadastro"
    st.session_state.tipo_selecionado = tipo
    st.session_state.modo_cadastro = "Cadastrar nova imagem"


# --- SIDEBAR ---
with st.sidebar:
    st.header("Vídeos de Apoio")
    st.video("https://youtu.be/hY5K55Ha2pg")
    st.write(f"Tutorial: {st.session_state.get('tipo_selecionado', 'Geral')}")

# --- CONTEÚDO DAS PÁGINAS ---
if st.session_state.pagina == "selecao":
    st.markdown(
        '<p class="titulo-amarelo">Selecione o tipo de análise:</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    if c1.button("Meia-Vida"):
        ir_para_cadastro("Meia-Vida")

    if c2.button("Granulometria"):
        ir_para_cadastro("Granulometria")

    if c3.button("Estabilidade"):
        ir_para_cadastro("Estabilidade Dinâmica")

elif st.session_state.pagina == "cadastro":
    st.subheader(f"Ficha de Cadastro: {st.session_state.tipo_selecionado}")

    if st.button("⬅️ Voltar ao Menu Principal"):
        st.session_state.pagina = "selecao"
        st.rerun()

    st.write("### Ação")

    modo = st.radio(
        "Escolha a ação",
        ["Cadastrar nova imagem", "Consultar imagem"],
        horizontal=True,
        key="modo_cadastro"
    )

    # ---------------------------------
    # MODO 1: CADASTRAR NOVA IMAGEM
    # ---------------------------------
    if modo == "Cadastrar nova imagem":
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

                        # Se quiser separar por tela daqui para frente:
                        # nome_final = f"granulometria/A{...}"
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
                        else:
                            st.error(f"Erro ao salvar no Supabase: {detalhe}")

    # ---------------------------------
    # MODO 2: CONSULTAR IMAGEM
    # ---------------------------------
    elif modo == "Consultar imagem":
        st.write("### Consultar imagem salva")

        termo_busca = st.text_input(
            "Buscar por nome do arquivo",
            placeholder="Ex.: A000, V08, 300, png..."
        )

        if st.button("Buscar imagens"):
            with st.spinner("Consultando imagens no Supabase..."):
                imagens, erro = listar_imagens_supabase(termo_busca)

            if erro:
                st.error(f"Erro ao listar imagens: {erro}")
            elif not imagens:
                st.warning("Nenhuma imagem encontrada.")
            else:
                opcoes = [img["name"] for img in imagens]

                st.session_state["lista_imagens_consulta"] = opcoes

        if "lista_imagens_consulta" in st.session_state and st.session_state["lista_imagens_consulta"]:
            arquivo_escolhido = st.selectbox(
                "Selecione a imagem",
                st.session_state["lista_imagens_consulta"]
            )

            if arquivo_escolhido:
                url_publica = montar_url_publica(arquivo_escolhido)

                st.write("### Pré-visualização")
                st.image(url_publica, caption=arquivo_escolhido, use_container_width=True)

                st.write("**Arquivo:**", arquivo_escolhido)
                st.write("**URL pública:**", url_publica)
