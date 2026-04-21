import streamlit as st
import base64
import io
fimport streamlit as st
from supabase import create_client

# --- CONFIGURAÇÃO E SUPABASE ---
def get_supabase_client():
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

# --- NOVA FUNÇÃO DE UPLOAD ---
def salvar_no_supabase(arquivo_bytes, nome_arquivo, mime_type):
    try:
        supabase = get_supabase_client()
        # 'imbfoam' é o nome do seu bucket
        response = supabase.storage.from_("imbfoam").upload(
            path=nome_arquivo,
            file=arquivo_bytes,
            file_options={"content-type": mime_type}
        )
        return True
    except Exception as e:
        st.error(f"Erro ao salvar no Supabase: {e}")
        return False


# --- CSS E ESTILIZAÇÃO ---
st.markdown("""
<style>
    #header-container {
        background-color: white;
        color: black;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        text-align: center;
    }

    .titulo-amarelo {
        color: #FFD700;
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }

    .card {
        background-color: #1E3A5F;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #2E7BCF;
        text-align: center;
        height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }

    .card h3 {
        color: #FFFFFF;
        font-size: 1.3em;
        margin: 0 0 10px 0;
    }

    .card p {
        color: #B9D1EA;
        font-size: 0.8em;
        margin: 0;
        line-height: 1.3;
    }
</style>
""", unsafe_allow_html=True)


# --- LÓGICA DE NAVEGAÇÃO ---
if "pagina" not in st.session_state:
    st.session_state.pagina = "selecao"


def ir_para_cadastro(tipo):
    st.session_state.pagina = "cadastro"
    st.session_state.tipo_selecionado = tipo


# --- SIDEBAR (Vídeos e Acessos) ---
with st.sidebar:
    st.header("Vídeos de Apoio")
    st.video("https://youtu.be/hY5K55Ha2pg")
    st.write(f"Tutorial: {st.session_state.get('tipo_selecionado', 'Geral')}")

    # A Central de Acessos aparece APENAS na seleção
    if st.session_state.pagina == "selecao":
        st.divider()
        st.subheader("🔑 Central de Acessos")
        with st.expander("Clique aqui para ver acessos"):
            st.markdown("""
            **E-mail**: `msbbfoam@gmail.com`  
            **Senha**: `Bfoam-50`
            """)


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
                # ... (Lógica de montagem do nome_final continua igual) ...

                with st.spinner("Enviando para o Supabase..."):
                    sucesso = salvar_no_supabase(
                        uploaded_file.getvalue(),
                        nome_final,
                        mime_type
                    )

                if sucesso:
                    st.success(f"Arquivo salvo com sucesso: {nome_final}")
                else:
                    st.error("Falha ao salvar o arquivo.")
