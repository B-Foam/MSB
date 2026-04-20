import streamlit as st
import base64 

# Configuração da Página
st.set_page_config(page_title="B-Foam MSB", page_icon="🔬", layout="centered")

def get_image_as_base64(path):
    try:
        with open(path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{data}"
    except: return ""

# --- CSS Omitido por brevidade, mantém o que você já tem ---
st.markdown("""<style>
    #header-container { background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 30px; display: flex; align-items: center; justify-content: center; width: 100%; text-align: center; }
    #header-container img { height: 60px; margin-right: 15px; }
    .titulo-amarelo { color: #FFD700; font-size: 4.0em; font-weight: bold; text-align: center; margin: 20px 0; }
    .card { background-color: #1E3A5F; padding: 15px; border-radius: 15px; border: 1px solid #2E7BCF; text-align: center; height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; }
    .card h3 { color: #FFFFFF; font-size: 1.3em; margin: 0 0 10px 0; }
    .card p { color: #B9D1EA; font-size: 0.8em; margin: 0; line-height: 1.3; }
</style>""", unsafe_allow_html=True)

if 'pagina' not in st.session_state: st.session_state.pagina = 'selecao'

def ir_para_cadastro(tipo):
    st.session_state.pagina = 'cadastro'
    st.session_state.tipo_selecionado = tipo

# --- FUNÇÃO PARA RENDERIZAR O SIDEBAR DINÂMICO ---
def renderizar_sidebar(url_video, legenda):
    with st.sidebar:
        st.header("Vídeos de Apoio")
        st.video(url_video)
        st.write(legenda)
        st.divider()
        st.subheader("🔑 Central de Acessos")
        with st.expander("Clique aqui para ver acessos"):
            st.markdown(f"**E-mail**: `msbbfoam@gmail.com`\n**Senha**: `Bfoam-50`")
        st.info("Dúvidas? Entre em contato via WhatsApp.")

# --- NAVEGAÇÃO ---
if st.session_state.pagina == 'selecao':
    renderizar_sidebar("https://youtu.be/hY5K55Ha2pg", "Assista ao vídeo geral de introdução.")
    
    logo = get_image_as_base64("logo-msb.png") 
    st.markdown(f'''<div id="header-container"><img src="{logo}">
                  <div><h1>B-Foam</h1><p>Engenharia MSB - Plataforma de Análise</p></div></div>''', unsafe_allow_html=True)
    
    st.markdown('<p class="titulo-amarelo">Selecione o tipo de análise desejada:</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.markdown('<div class="card"><h3>Meia-Vida</h3><p>Análise de decaimento.</p></div>', unsafe_allow_html=True)
        if st.button("Selecionar", key="mv"): ir_para_cadastro("Meia-Vida")
    with c2: 
        st.markdown('<div class="card"><h3>Granulometria</h3><p>Medição de bolhas.</p></div>', unsafe_allow_html=True)
        if st.button("Selecionar", key="gr"): ir_para_cadastro("Granulometria")
    with c3: 
        st.markdown('<div class="card"><h3>Estabilidade</h3><p>Resistência estrutural.</p></div>', unsafe_allow_html=True)
        if st.button("Selecionar", key="ed"): ir_para_cadastro("Estabilidade Dinâmica")

elif st.session_state.pagina == 'cadastro':
    # AQUI VOCÊ DEFINE O VÍDEO ESPECÍFICO PARA A PÁGINA DE CADASTRO
    renderizar_sidebar("https://youtu.be/OUTRO_LINK_VIDEO", f"Vídeo tutorial para: {st.session_state.tipo_selecionado}")
    
    st.subheader(f"Ficha de Cadastro: {st.session_state.tipo_selecionado}")
    if st.button("⬅️ Voltar ao Menu Principal"):
        st.session_state.pagina = 'selecao'
        st.rerun()
    
    tab1, _ = st.tabs(["➕ Cadastrar Novo Teste", "🔍 Buscar Histórico"])
    with tab1:
        # ... (seu código do formulário de cadastro que já fizemos)
        st.write("Formulário de cadastro ativo.")
