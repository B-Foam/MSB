import streamlit as st
import base64
import io
import json
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload




# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="B-Foam MSB", page_icon="🔬", layout="centered")

# --- FUNÇÃO PARA SALVAR NO DRIVE (SEGURA) ---
def get_drive_service():
    # Carrega a string JSON do segredo 'key' e transforma em dicionário
    secret_data = st.secrets["gcp_service_account"]
    creds_dict = json.loads(secret_data["key"])
    
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/drive.file']
    )
    return build('drive', 'v3', credentials=creds)

def salvar_no_drive(arquivo_bytes, nome_arquivo):
    """
    Salva um arquivo bytes no Google Drive em uma pasta específica.
    """
    try:
        # 1. Obter serviço autenticado
        creds_dict = json.loads(st.secrets["gcp_service_account"]["key"])
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=['https://www.googleapis.com/auth/drive.file']
        )
        service = build('drive', 'v3', credentials=creds)
        
        # 2. ID da pasta (Substitua pelo ID que você pegou na URL do seu Drive)
        folder_id = 'https://drive.google.com/drive/u/6/folders/1JhYTZ74AYLu0FZAoaE4mb-nD7IFYAP5i' 
        
        # 3. Metadados do arquivo
        file_metadata = {
            'name': nome_arquivo,
            'parents': [folder_id]
        }
        
        # 4. Upload
        media = MediaIoBaseUpload(io.BytesIO(arquivo_bytes), mimetype='image/png')
        file = service.files().create(
            body=file_metadata, 
            media_body=media, 
            fields='id'
        ).execute()
        
        return file.get('id')
        
    except Exception as e:
        st.error(f"Erro ao comunicar com o Google Drive: {e}")
        return None

def get_image_as_base64(path):
    try:
        with open(path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{data}"
    except: return ""

# --- CSS E ESTILIZAÇÃO ---
st.markdown("""
<style>
    #header-container { background-color: white; color: black; padding: 20px; border-radius: 10px; margin-bottom: 30px; display: flex; align-items: center; justify-content: center; width: 100%; text-align: center; }
    .titulo-amarelo { color: #FFD700; font-size: 3.5em; font-weight: bold; text-align: center; margin: 20px 0; }
    .card { background-color: #1E3A5F; padding: 15px; border-radius: 15px; border: 1px solid #2E7BCF; text-align: center; height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; }
    .card h3 { color: #FFFFFF; font-size: 1.3em; margin: 0 0 10px 0; }
    .card p { color: #B9D1EA; font-size: 0.8em; margin: 0; line-height: 1.3; }
</style>
""", unsafe_allow_html=True)

# --- LÓGICA DE NAVEGAÇÃO ---
if 'pagina' not in st.session_state: st.session_state.pagina = 'selecao'

def ir_para_cadastro(tipo):
    st.session_state.pagina = 'cadastro'
    st.session_state.tipo_selecionado = tipo

# --- SIDEBAR (Vídeos e Acessos) ---
with st.sidebar:
    st.header("Vídeos de Apoio")
    st.video("https://youtu.be/hY5K55Ha2pg")
    st.write(f"Tutorial: {st.session_state.get('tipo_selecionado', 'Geral')}")
    
    # A Central de Acessos aparece APENAS na seleção
    if st.session_state.pagina == 'selecao':
        st.divider()
        st.subheader("🔑 Central de Acessos")
        with st.expander("Clique aqui para ver acessos"):
            st.markdown("""
            **E-mail**: `msbbfoam@gmail.com`  
            **Senha**: `Bfoam-50`
            """)

# --- CONTEÚDO DAS PÁGINAS ---
if st.session_state.pagina == 'selecao':
    st.markdown('<p class="titulo-amarelo">Selecione o tipo de análise:</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if c1.button("Meia-Vida"): ir_para_cadastro("Meia-Vida")
    if c2.button("Granulometria"): ir_para_cadastro("Granulometria")
    if c3.button("Estabilidade"): ir_para_cadastro("Estabilidade Dinâmica")

elif st.session_state.pagina == 'cadastro':
    st.subheader(f"Ficha de Cadastro: {st.session_state.tipo_selecionado}")
    if st.button("⬅️ Voltar ao Menu Principal"):
        st.session_state.pagina = 'selecao'
        st.rerun()
    
    with st.form("form_novo_teste", clear_on_submit=True):
        amostra = st.text_input("Amostra (ex: 001)")
        teste = st.text_input("Teste (ex: 001)")
        tempo = st.number_input("Tempo de estabilidade (segundos)", min_value=0, step=1)
        concentracao = st.selectbox("Concentração do Polidocanol", ["3,00%", "1,00%", "0,50%", "0,25%"])
        dispositivo = st.selectbox("Dispositivo utilizado", ["V08", "V09", "V10", "Tessari", "Outros"])
        
        outro_dispositivo = st.text_input("Especifique o dispositivo:") if dispositivo == "Outros" else ""
        uploaded_file = st.file_uploader("Escolha a imagem do teste:", type=['png', 'jpg', 'jpeg'])
        
        if st.form_submit_button("Salvar Registro no Drive"):
            if uploaded_file is not None:
                # Lógica da codificação
                c_clean = concentracao.replace(',', '').replace('%', '')
                disp_final = outro_dispositivo if dispositivo == "Outros" else dispositivo
                nome_final = f"A{amostra.zfill(3)}_T{teste.zfill(3)}_{tempo}s_{c_clean}_{disp_final}.png"
                
                with st.spinner('Salvando no Drive...'):
                    try:
                        file_id = salvar_no_drive(uploaded_file.getvalue(), nome_final)
                        st.success(f"Arquivo salvo com sucesso! Nome: {nome_final}")
                    except Exception as e:
                        st.error(f"Erro ao salvar: {e}")
            else:
                st.error("Por favor, faça o upload da imagem primeiro..")
