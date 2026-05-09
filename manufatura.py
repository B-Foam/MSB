import streamlit as st
from pathlib import Path


def render_revista_msb():
    st.markdown("## 📘 Revista Técnica MSB")
    st.markdown(
        """
        Guia visual de materiais utilizados em dispositivos médicos, com foco em polímeros,
        elastômeros, metais, cerâmicas, adesivos, revestimentos, radiopacificantes e normas.
        """
    )

    revista_dir = Path("assets/revista")

    paginas = [
        {"titulo": "Capa", "arquivo": "01-capa.png"},
        {"titulo": "Página em branco", "arquivo": "02-pagina-branca.png"},
        {"titulo": "Sumário", "arquivo": "03-sumario.png"},
        {"titulo": "Polímeros e Termoplásticos", "arquivo": "04-polimeros.png"},
        {"titulo": "Elastômeros e Borrachas", "arquivo": "05-elastomeros.png"},
        {"titulo": "Metais e Ligas", "arquivo": "06-metais.png"},
        {"titulo": "Cerâmicas Avançadas", "arquivo": "07-ceramicas.png"},
        {"titulo": "Adesivos e Revestimentos", "arquivo": "08-adesivos.png"},
        {"titulo": "Radiopacificantes e Aditivos", "arquivo": "09-radiopacificantes.png"},
        {"titulo": "Normas e Certificações", "arquivo": "10-normas.png"},
        {"titulo": "Contracapa", "arquivo": "11-contracapa.png"},
    ]

    if "pagina_revista_msb" not in st.session_state:
        st.session_state.pagina_revista_msb = 0

    total_paginas = len(paginas)

    if st.session_state.pagina_revista_msb < 0:
        st.session_state.pagina_revista_msb = 0

    if st.session_state.pagina_revista_msb > total_paginas - 1:
        st.session_state.pagina_revista_msb = total_paginas - 1

    pagina_atual = st.session_state.pagina_revista_msb
    pagina_info = paginas[pagina_atual]
    caminho_imagem = revista_dir / pagina_info["arquivo"]

    st.markdown(
        """
        <style>
        .revista-container {
            background: linear-gradient(135deg, #f7fbff 0%, #ffffff 50%, #eaf4fb 100%);
            border-radius: 22px;
            padding: 24px;
            border: 1px solid #d9e8f2;
            box-shadow: 0 8px 28px rgba(0, 35, 70, 0.08);
            margin-top: 18px;
            margin-bottom: 18px;
        }

        .revista-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            color: #0b2d55;
            font-weight: 700;
            font-size: 18px;
        }

        .revista-subtitle {
            color: #4b6b88;
            font-size: 14px;
            margin-top: -8px;
            margin-bottom: 16px;
        }

        .revista-footer {
            text-align: center;
            color: #4b6b88;
            font-size: 13px;
            margin-top: 12px;
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid #0b5ea8;
            color: #0b2d55;
            font-weight: 600;
        }

        .stButton > button:hover {
            border-color: #0077c8;
            color: #0077c8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="revista-container">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="revista-header">
            <div>{pagina_info["titulo"]}</div>
            <div>Página {pagina_atual + 1} de {total_paginas}</div>
        </div>
        <div class="revista-subtitle">
            MSB Medical System do Brasil — Engenharia e Materiais para Dispositivos Médicos
        </div>
        """,
        unsafe_allow_html=True,
    )

    if caminho_imagem.exists():
        st.image(str(caminho_imagem), use_container_width=True)
    else:
        st.error(f"Imagem não encontrada: {caminho_imagem}")
        st.warning(
            "Verifique se o nome do arquivo está igual ao nome salvo no GitHub "
            "e se ele está dentro da pasta assets/revista."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("⏮️ Início", use_container_width=True, key="btn_revista_inicio_msb"):
            st.session_state.pagina_revista_msb = 0
            st.rerun()

    with col2:
        if st.button("⬅️ Anterior", use_container_width=True, key="btn_revista_anterior_msb"):
            if st.session_state.pagina_revista_msb > 0:
                st.session_state.pagina_revista_msb -= 1
            st.rerun()

    with col3:
        pagina_selecionada = st.selectbox(
            "Ir para a página:",
            options=list(range(total_paginas)),
            format_func=lambda i: f"{i + 1:02d} - {paginas[i]['titulo']}",
            index=pagina_atual,
            label_visibility="collapsed",
            key="select_pagina_revista_msb",
        )

        if pagina_selecionada != pagina_atual:
            st.session_state.pagina_revista_msb = pagina_selecionada
            st.rerun()

    with col4:
        if st.button("Próxima ➡️", use_container_width=True, key="btn_revista_proxima_msb"):
            if st.session_state.pagina_revista_msb < total_paginas - 1:
                st.session_state.pagina_revista_msb += 1
            st.rerun()

    with col5:
        if st.button("Fim ⏭️", use_container_width=True, key="btn_revista_fim_msb"):
            st.session_state.pagina_revista_msb = total_paginas - 1
            st.rerun()

    st.markdown(
        f"""
        <div class="revista-footer">
            Visualizando: <strong>{pagina_info["arquivo"]}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_manufatura():
    st.markdown("# 🏭 Manufatura")

    if st.button("⬅️ Voltar ao Menu Principal", key="btn_voltar_menu_manufatura"):
        st.session_state.pagina = "selecao"
        st.rerun()

    st.markdown(
        """
        Esta área será destinada ao acompanhamento das informações relacionadas à manufatura,
        incluindo registros, controles, documentos, dados de apoio ao processo produtivo
        e materiais técnicos utilizados no desenvolvimento de dispositivos médicos.
        """
    )

    st.divider()

    aba1, aba2, aba3, aba4 = st.tabs(
        [
            "📋 Registros",
            "⚙️ Processos",
            "📁 Documentos",
            "📘 Revista MSB",
        ]
    )

    with aba1:
        st.subheader("Registros de Manufatura")
        st.info("Área reservada para cadastro e consulta de registros de manufatura.")

        st.markdown(
            """
            Sugestões para esta área:

            - Registro de lote;
            - Registro de produção;
            - Controle de matéria-prima;
            - Controle de inspeção;
            - Histórico de alterações;
            - Rastreabilidade de componentes.
            """
        )

    with aba2:
        st.subheader("Processos de Fabricação")
        st.info("Área reservada para acompanhamento dos processos produtivos.")

        st.markdown(
            """
            Sugestões para esta área:

            - Injeção plástica;
            - Extrusão;
            - Usinagem;
            - Soldagem;
            - Colagem;
            - Montagem;
            - Embalagem;
            - Esterilização.
            """
        )

    with aba3:
        st.subheader("Documentos de Manufatura")
        st.info("Área reservada para documentos, instruções e arquivos relacionados.")

        st.markdown(
            """
            Sugestões para esta área:

            - Instruções de trabalho;
            - Procedimentos operacionais;
            - Fichas técnicas;
            - Certificados de matéria-prima;
            - Validações;
            - Registros de qualidade.
            """
        )

    with aba4:
        render_revista_msb()
