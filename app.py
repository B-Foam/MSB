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

st.markdown("""
<style>
    /* Ajustes Gerais */
    #header-container {
        background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 30px;
        display: flex; align-items: center; justify-content: center; width: 100%; text-align: center;
    }
    #header-container img { height: 60px; margin-right: 15px; }
    
    .titulo-amarelo { color: #FFD700; font-size: 4.0em; font-weight: bold; text-align: center; margin: 20px 0; }

    /* Cards Centralizados */
    .card { 
        background-color: #1E3A5F; padding: 15px; border-radius: 15px; 
        border: 1px solid #2E7BCF; text-align: center; height: 200px;
        display: flex; flex-direction: column; align-items: center; justify-content: flex-start;
    }
    .card h3 { color: #FFFFFF; font-size: 1.3em; margin: 0 0 10px 0; } /* Fonte levemente reduzida */
    .card p { color: #B9D1EA; font-size: 0.8em; margin: 0; line-height: 1.3; }

       /* Botão Centralizado e com tamanho adaptável */
    div.stButton { 
        display: flex; 
        justify-content: center;
        margin-top: 10px;
        width: 100%; /* Garante que o contêiner ocupe a coluna */
    }

    div.stButton > button { 
        /* Usamos min-width para garantir que ele não fique minúsculo, 
           mas permitimos que cresça se necessário */
        min-width: 120px; 
        padding: 5px 15px;    
        border-radius: 8px; 
        border: 1px solid #2E7BCF;
        white-space: nowrap;   /* Impede que o texto quebre em duas linhas */
        font-size: 0.9em;      /* Fonte um pouco menor para caber melhor */
    }
</style>
""", unsafe_allow_html=True)

if 'pagina' not in st.session_state: st.session_state.pagina = 'selecao'
def ir_para_cadastro(tipo):
    st.session_state.pagina = 'cadastro'
    st.session_state.tipo_selecionado = tipo

if st.session_state.pagina == 'selecao':
    logo = get_image_as_base64("logo-msb.png") 
    st.markdown(f'''<div id="header-container"><img src="{logo}">
                  <div><h1>B-Foam</h1><p>Engenharia MSB - Plataforma de Análise</p></div></div>''', unsafe_allow_html=True)

    st.markdown('<p class="titulo-amarelo">Selecione o tipo de análise desejada:</p>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    # Cards sem o símbolo de link
    with c1:
        st.markdown('<div class="card"><h3>Meia-Vida</h3><p>Análise do tempo de decaimento das microbolhas para determinar a longevidade da espuma.</p></div>', unsafe_allow_html=True)
        if st.button("Selecionar", key="mv"): ir_para_cadastro("Meia-Vida")
    with c2:
        st.markdown('<div class="card"><h3>Granulometria</h3><p>Medição do tamanho e distribuição das bolhas para avaliar a homogeneidade da amostra.</p></div>', unsafe_allow_html=True)
        if st.button("Selecionar", key="gr"): ir_para_cadastro("Granulometria")
    with c3:
        st.markdown('<div class="card"><h3>Estabilidade</h3><p>Avaliação da resistência estrutural da espuma sob variações de pressão e tempo.</p></div>', unsafe_allow_html=True)
        if st.button("Selecionar", key="ed"): ir_para_cadastro("Estabilidade Dinâmica")

    with st.sidebar:
        st.header("Vídeos de Apoio")
        st.video("https://youtu.be/hY5K55Ha2pg")
        st.write("Assista ao teste de medição do tamanho de bolhas.")

    # --- NOVO BLOCO DE ACESSOS ---
        st.divider()
        st.subheader("🔑 Central de Acessos")
        
        with st.expander("Clique aqui para ver acessos"):
            st.markdown(f"""
            **E-mail**: `msbbfoam@gmail.com`  
            **Senha**: `Bfoam-50`
            
            ---
            **Links Úteis**:
            * [🔗 Google Drive](https://drive.google.com)
            * [🔗 GitHub](https://github.com/B-Foam/MSB)
            * [🔗 Streamlit Cloud](https://share.streamlit.io)
            """)
        
        st.info("Dúvidas? Entre em contato via WhatsApp para suporte técnico.")

elif st.session_state.pagina == 'cadastro':
    st.subheader(f"Ficha de Cadastro: {st.session_state.tipo_selecionado}")
    if st.button("⬅️ Voltar ao Menu Principal"):
        st.session_state.pagina = 'selecao'
        st.rerun()
    tab1, tab2 = st.tabs(["➕ Cadastrar Novo Teste", "🔍 Buscar Histórico"])
    with tab1:
        st.markdown("### 📸 Upload e Identificação do Teste")
        
        # Criação de duas colunas para organização
        col_img, col_form = st.columns([1, 1.5])
        
        with col_img:
            uploaded_file = st.file_uploader("Escolha a imagem do teste:", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                st.image(uploaded_file, caption="Imagem carregada", use_column_width=True)

        with col_form:
            with st.form("form_novo_teste", clear_on_submit=True):
                # Campos de texto
                amostra = st.text_input("Amostra (ex: 1)", placeholder="001")
                teste = st.text_input("Teste (ex: 1)", placeholder="001")
                tempo = st.number_input("Tempo de estabilidade (segundos)", min_value=0, step=1)
                
                # Campos de seleção
                concentracao = st.selectbox("Concentração do Polidocanol", 
                                            ["3,00%", "1,00%", "0,50%", "0,25%"])
                
                dispositivo = st.selectbox("Dispositivo utilizado", 
                                          ["V08", "V09", "V10", "Tessari", "Outros"])
                
                # Lógica para o campo "Outros"
                outro_dispositivo = ""
                if dispositivo == "Outros":
                    outro_dispositivo = st.text_input("Especifique o outro dispositivo:")

                # Botão de Salvar
                btn_salvar = st.form_submit_button("Salvar Registro")

                if btn_salvar:
                    # Lógica de padronização (transforma "1" em "001")
                    amostra_fmt = amostra.zfill(3)
                    teste_fmt = teste.zfill(3)
                    disp_final = outro_dispositivo if dispositivo == "Outros" else dispositivo
                    
                    if uploaded_file:
                        # Nome do arquivo conforme seus critérios
                        nome_arquivo = f"A{amostra_fmt}_T{teste_fmt}_{tempo}s_{concentracao}_{disp_final}.png"
                        st.success(f"Dados validados! Pronto para salvar como: {nome_arquivo}")
                        # DICA: Aqui você chamaria sua função de salvar o arquivo em disco/banco
                    else:
                        st.error("Por favor, faça o upload da imagem antes de salvar.")
