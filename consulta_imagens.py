import streamlit as st


def render_consulta_imagens(
    listar_imagens_supabase,
    montar_url_publica,
    session_state
):
    with st.container(border=True):
        st.subheader("Consultar imagens")

        termo_busca = st.text_input("Buscar por nome do arquivo")

        if st.button("Atualizar lista", key="btn_atualizar_lista_consulta"):
            imagens, erro = listar_imagens_supabase(termo_busca)

            if erro:
                st.error(f"Erro: {erro}")
            else:
                session_state.lista_imagens_consulta = [img["name"] for img in imagens]

        lista = session_state.get("lista_imagens_consulta", [])

        if lista:
            escolhido = st.selectbox(
                "Selecione a imagem",
                lista,
                key="select_imagem_consulta"
            )

            if escolhido:
                st.image(
                    montar_url_publica(escolhido),
                    use_container_width=True
                )
        else:
            st.info("Nenhuma imagem carregada ainda.")
