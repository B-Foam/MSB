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

# --- CSS ---
st.markdown("""
<style>
    #header-container { background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 30px; display: flex; align-items: center; justify-content: center; width: 100%; text-align: center; }
    #header-container img { height: 60px; margin-right: 15px; }
    .titulo-amarelo { color: #FFD700; font-size: 4.0em; font-weight: bold; text-align: center; margin: 20px 0; }
    .card { background-color: #1E3A5F; padding: 15px; border-radius: 15px; border: 1px solid #2E7BCF; text-align: center; height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; }
    .card h3 { color: #FFFFFF; font-size: 1.3em; margin: 0 0 10px 0; }
    .card p { color: #B9D1EA; font-size: 0.8em; margin: 0; line-height: 1.3; }
    div.stButton { display: flex; justify-content: center; margin-top: 10px; width: 100%; }
    div.stButton > button { min-width: 120px; padding: 5px 15px; border-radius: 8px; border: 1px solid #2E7BCF; white-space: nowrap; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# --- NAVEGAÇÃO ---
if 'pagina' not in st.session_state: st.session_state.pagina = 'selecao'

def ir_para_cadastro(tipo):
    st.session_state.pagina = 'cadastro'
    st.session_state.tipo_selecionado = tipo

# --- PÁGINA DE SELEÇÃO ---
if st.session_state.pagina == 'selecao':
    logo = get_image_as_base64("logo-msb.png") 
    st.markdown(f'''<div id="header-container"><img src="{logo}">
                  <div><h1>B-Foam</h1><p>Engenharia MSB - Plataforma de Análise</p></div></div>''', unsafe_allow_html=True)
    
    st.markdown('<p class="titulo-amarelo">Selecione o tipo de análise desejada:</p>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h3>Meia-Vida</h3><p>Análise do tempo de decaimento.</p></div>', unsafe_allow_html=True)
        if st.button("Selecionar", key="mv"): ir_para_cadastro("Meia-Vida")
    with c2:
        st.markdown('<div class="card"><h3>Granulometria</h3><p>Medição do tamanho e distribuição das bolhas.</p></div>', unsafe_allow_html=True)
        if st.button("Selecionar", key="gr"): ir_para_cadastro("Granulometria")
    with c3:
        st.markdown('<div class="card"><h3>Estabilidade</h3><p>Avaliação da resistência estrutural.</p></div>', unsafe_allow_html=True)
        if st.button("Selecionar", key="ed"): ir_para_cadastro("Estabilidade Dinâmica")

# --- PÁGINA DE CADASTRO ---
elif st.session_state.pagina == 'cadastro':
    # A SIDEBAR SÓ APARECE SE O TIPO FOR GRANULOMETRIA
    if st.session_state.tipo_selecionado == "Granulometria":
        with st.sidebar:
            st.header("Vídeos de Apoio")
            st.video("https://youtu.be/hY5K55Ha2pg")
            st.write("Tutorial técnico: Granulometria")
    
    st.subheader(f"Ficha de Cadastro: {st.session_state.tipo_selecionado}")
    
    if st.button("⬅️ Voltar ao Menu Principal"):
        st.session_state.pagina = 'selecao'
        st.rerun()
    
    tab1, tab2 = st.tabs(["➕ Cadastrar Novo Teste", "🔍 Buscar Histórico"])
    with tab1:
        st.markdown("### 📸 Upload e Identificação do Teste")
        col_img, col_form = st.columns([1, 1.5])
        
        with col_img:
            uploaded_file = st.file_uploader("Escolha a imagem do teste:", type=['png', 'jpg', 'jpeg'])
            if uploaded_file: st.image(uploaded_file, caption="Imagem carregada", use_column_width=True)

        with col_form:
            with st.form("form_novo_teste", clear_on_submit=True):
                amostra = st.text_input("Amostra (ex: 1)", placeholder="001")
                teste = st.text_input("Teste (ex: 1)", placeholder="001")
                tempo = st.number_input("Tempo de estabilidade (segundos)", min_value=0, step=1)
                concentracao = st.selectbox("Concentração do Polidocanol", ["3,00%", "1,00%", "0,50%", "0,25%"])
                dispositivo = st.selectbox("Dispositivo utilizado", ["V08", "V09", "V10", "Tessari", "Outros"])
                
                outro_dispositivo = st.text_input("Especifique o dispositivo:") if dispositivo == "Outros" else ""
                
                if st.form_submit_button("Salvar Registro"):
                    if uploaded_file:
                        nome = f"A{amostra.zfill(3)}_T{teste.zfill(3)}_{tempo}s_{concentracao}_{outro_dispositivo if dispositivo == 'Outros' else dispositivo}.png"
                        st.success(f"Dados validados! Pronto para salvar como: {nome}")
                    else:
                        st.error("Por favor, faça o upload da imagem.")
