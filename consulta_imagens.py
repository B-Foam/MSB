import io
import requests
import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


def baixar_imagem(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def filtro_tratamento_1(img):
    """
    Filtro 1:
    - escala de cinza
    - aumento de contraste automático
    - leve nitidez
    Objetivo: destacar melhor contornos e diferenças de intensidade.
    """
    img_gray = ImageOps.grayscale(img)
    img_auto = ImageOps.autocontrast(img_gray, cutoff=1)
    img_sharp = img_auto.filter(ImageFilter.SHARPEN)
    return img_sharp


def filtro_tratamento_2(img):
    """
    Filtro 2:
    - escala de cinza
    - redução de ruído por mediana
    - aumento de contraste
    - reforço de borda
    Objetivo: suavizar ruído e deixar bolhas mais evidentes.
    """
    img_gray = ImageOps.grayscale(img)
    img_med = img_gray.filter(ImageFilter.MedianFilter(size=3))
    img_contrast = ImageEnhance.Contrast(img_med).enhance(1.8)
    img_edge = img_contrast.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return img_edge


def render_consulta_imagens(
    listar_imagens_supabase,
    montar_url_publica,
    session_state
):
    with st.container(border=True):
        if st.button("Atualizar lista", key="btn_atualizar_lista_consulta"):
            imagens, erro = listar_imagens_supabase("")

            if erro:
                st.error(f"Erro: {erro}")
            else:
                session_state.lista_imagens_consulta = [img["name"] for img in imagens]

        lista = session_state.get("lista_imagens_consulta", [])

        if not lista:
            st.info("Clique em 'Atualizar lista' para carregar as imagens.")
            return

        escolhido = st.selectbox(
            "Selecione a imagem",
            lista,
            key="select_imagem_consulta"
        )

        if not escolhido:
            return

        url_imagem = montar_url_publica(escolhido)

        try:
            img_original = baixar_imagem(url_imagem)
            img_filtro_1 = filtro_tratamento_1(img_original)
            img_filtro_2 = filtro_tratamento_2(img_original)

            st.markdown("### Pré-visualização para tratamento")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.caption("Imagem original")
                st.image(img_original, use_container_width=True)

            with col2:
                st.caption("Filtro 1 — contraste e nitidez")
                st.image(img_filtro_1, use_container_width=True)

            with col3:
                st.caption("Filtro 2 — redução de ruído e bordas")
                st.image(img_filtro_2, use_container_width=True)

            st.markdown("### Observação")
            st.write(
                "Esses dois filtros são uma etapa inicial para preparar a imagem "
                "para futuras rotinas de calibração, contagem e medição de bolhas."
            )

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
