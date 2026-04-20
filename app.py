import streamlit as st
import base64
import io
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload

# Configuração da Página
st.set_page_config(page_title="B-Foam MSB", page_icon="🔬", layout="centered")

# --- FUNÇÕES DE APOIO ---
def get_image_as_base64(path):
    try:
        with open(path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{data}"
    except: return ""

def salvar_no_drive(arquivo_bytes, nome_arquivo):
    # Ajuste o caminho do arquivo JSON se necessário
    creds = service_account.Credentials.from_service_account_file('credenciais.json', 
            scopes=['https://www.googleapis.com/auth/drive.file'])
    service = build('drive', 'v3', credentials=creds)
    
    file_metadata = {'name': nome_arquivo, 'parents': ['ID_DA_SUA_PASTA_NO_DRIVE']}
    media = MediaIoBaseUpload(io.BytesIO(arquivo_bytes), mimetype='image/png')
    
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# --- CSS ---
st.markdown("""
<style>
    #header-container { background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 30px; display: flex; align-items: center; justify-content: center; width: 100%; text-align: center; }
    .titulo-amarelo { color: #FFD700; font-size: 4.0em; font-weight: bold; text-align: center; margin: 20px 0; }
    .card { background-color: #1E3A5F; padding: 15px; border-radius: 15px; border: 1px solid #2E7BCF; text-align: center; height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; }
</style>
""", unsafe_allow_html=True)

# --- NAVEGAÇÃO ---
if 'pagina' not in st.session_state: st.session_state.pagina = 'selecao'

def ir_para_cadastro(tipo):
    st.session_state.pagina = 'cadastro'
    st.session_state.tipo_selecionado = tipo

# --- SIDEBAR ---
with st.sidebar:
    st.header("Vídeos de Apoio")
    st.video("https://youtu.be/hY5K55Ha2pg")
    st.write(f"Tutorial: {st.session_state.get('tipo_selecionado', 'Geral')}")
    
    if st.session_state.pagina == 'selecao':
        st.divider()
        st.subheader("🔑 Central de Acessos")
        with st.expander("Clique aqui para ver acessos"):
            st.markdown("**E-mail**: `msbbfoam@gmail.com`\n**Senha**: `Bfoam-50`")

# --- CONTEÚDO ---
if st.session_state.pagina == 'selecao':
    st.markdown('<p class="titulo-amarelo">Selecione o tipo de análise:</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("Meia-Vida"): ir_para_cadastro("Meia-Vida")
    if c2.button("Granulometria"): ir_para_cadastro("Granulometria")
    if c3.button("Estabilidade"): ir_para_cadastro("Estabilidade Dinâmica")

elif st.session_state.pagina == 'cadastro':
    st.subheader(f"Ficha: {st.session_state.tipo_selecionado}")
    if st.button("⬅️ Voltar"): st.session_state.pagina = 'selecao'; st.rerun()
    
    with st.form("form_teste", clear_on_submit=True):
        amostra = st.text_input("Amostra (ex: 001)")
        teste = st.text_input("Teste (ex: 001)")
        tempo = st.number_input("Tempo (segundos)", min_value=0)
        concentracao = st.selectbox("Conc.", ["3,00%", "1,00%", "0,50%", "0,25%"])
        dispositivo = st.selectbox("Disp.", ["V08", "V09", "V10", "Tessari"])
        uploaded_file = st.file_uploader("Imagem", type=['png', 'jpg'])
        
        if st.form_submit_button("Salvar no Drive"):
            if uploaded_file:
                # Lógica da codificação: A001_001_30s_300_V08.png
                conc_limpa = concentracao.replace(',', '').replace('%', '')
                nome = f"A{amostra.zfill(3)}_{teste.zfill(3)}_{tempo}s_{conc_limpa}_{dispositivo}.png"
                
                try:
                    file_id = salvar_no_drive(uploaded_file.getvalue(), nome)
                    st.success(f"Salvo no Drive! ID: {file_id}")
                except Exception as e:
                    st.error(f"Erro: {e}")
