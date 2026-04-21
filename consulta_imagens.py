import io
import requests
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# ... (funções baixar_imagem, pil_to_cv, cv_to_pil permanecem iguais) ...

def detectar_barra_escala_px(img_bgr):
    """
    Detecta a barra horizontal preta no canto inferior esquerdo.
    Retorna o comprimento em pixels da barra horizontal e a imagem com a barra anotada.
    """
    h, w = img_bgr.shape[:2]

    # Região de interesse (ROI) mais focada no canto inferior esquerdo
    crop = img_bgr[int(h * 0.85):h, 0:int(w * 0.25)].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Pré-processamento para destacar a barra preta: binarização adaptativa
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)

    # Limpeza morfológica para remover ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor_largura = None
    melhor_cnt = None

    for cnt in contours:
        x_c, y_c, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / max(hh, 1)

        # Critérios para identificar a barra: horizontal, fina, comprimento mínimo
        if ww > 50 and hh < 15 and aspect > 8:
            if melhor_largura is None or ww > melhor_largura:
                melhor_largura = ww
                melhor_cnt = cnt

    # Criar imagem de feedback da calibração
    img_calibracao = crop.copy()
    if melhor_cnt is not None:
        # Desenhar o contorno detectado em verde na ROI
        cv2.drawContours(img_calibracao, [melhor_cnt], -1, (0, 255, 0), 2)
        
    return melhor_largura, img_calibracao


# ... (funções aplicar_filtro_1, aplicar_filtro_2, mascarar_regioes_fixas, remover_circulos_duplicados permanecem iguais) ...

def detectar_bolhas_hough(
    img_gray,
    dp,
    min_dist,
    param1,
    param2,
    min_radius,
    max_radius
):
    # Aplicar um leve sharpening antes da detecção pode ajudar
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_sharpened = cv2.filter2D(img_gray, -1, kernel_sharpen)

    circles = cv2.HoughCircles(
        img_sharpened,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype(int)
    return circles.tolist()

# ... (outras funções auxiliares permanecem iguais) ...

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

        # Estado para armazenar a calibração validada
        if "px_per_mm_validado" not in session_state:
            session_state.px_per_mm_validado = None

        escolhido = st.selectbox(
            "Selecione a imagem",
            lista,
            key="select_imagem_consulta"
        )

        if not escolhido:
            return

        url_imagem = montar_url_publica(escolhido)

        try:
            img_original_pil = baixar_imagem(url_imagem)
            img_original_bgr = pil_to_cv(img_original_pil)
            img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB) # Para exibição no Streamlit

            st.markdown("### 1. Calibração da Escala")
            
            # Detecção automática da barra de escala
            barra_px, img_calibracao_bgr = detectar_barra_escala_px(img_original_bgr)
            img_calibracao_rgb = cv2.cvtColor(img_calibracao_bgr, cv2.COLOR_BGR2RGB)

            colc1, colc2 = st.columns([1, 2])

            with colc1:
                st.image(img_calibracao_rgb, caption="Detecção da barra (ROI)", use_container_width=True)
                if barra_px is not None:
                    st.success(f"Barra de 1,0 mm detectada automaticamente: {barra_px} px")
                else:
                    st.warning("Barra não detectada automaticamente na região esperada.")

            with colc2:
                # Entrada manual para px por mm, pré-preenchida se detectado automaticamente
                px_per_mm_input = st.number_input(
                    "Pixels por milímetro (px/mm)",
                    min_value=1.0,
                    value=float(barra_px) if barra_px is not None else 100.0,
                    step=0.1,
                    help="Ajuste este valor manualmente se a detecção automática falhar. Verifique a imagem de feedback ao lado."
                )
                
                if st.button("Validar Calibração"):
                    session_state.px_per_mm_validado = px_per_mm_input
                    st.success(f"Calibração validada: {px_per_mm_input:.1f} px/mm")

            # Só prossegue se a calibração estiver validada
            if session_state.px_per_mm_validado is None:
                st.info("Por favor, valide a calibração da escala acima para prosseguir com a detecção de bolhas.")
                return

            px_per_mm = session_state.px_per_mm_validado

            st.markdown("---")
            st.markdown("### 2. Parâmetros de Detecção de Bolhas")
            
            # Novos valores padrão sugeridos para os sliders, baseados nas imagens
            colp1, colp2, colp3 = st.columns(3)
            with colp1:
                dp = st.slider("dp (resolução)", 1.0, 3.0, 1.5, 0.1, help="Maior dp = menor resolução, detecção mais rápida.")
                min_dist = st.slider("Distância mínima entre centros (px)", 5, 100, 15, 1)
            with colp2:
                param1 = st.slider("param1 (Canny superior)", 20, 300, 70, 5, help="Limiar superior para detecção de bordas.")
                param2 = st.slider("param2 (Limiar de detecção)", 5, 100, 15, 1, help="Menor valor = detecta mais bolhas, mas aumenta falsos positivos.")
            with colp3:
                min_radius = st.slider("Raio mínimo (px)", 2, 100, 5, 1)
                max_radius = st.slider("Raio máximo (px)", 20, 500, 120, 5)

            # Filtros e Detecção
            # Aplicar filtro 2 (suavização + contraste) parece ser o melhor ponto de partida
            img_suavizada_rgb = cv2.cvtColor(aplicar_filtro_2(img_original_bgr), cv2.COLOR_GRAY2RGB)
            img_processada_gray = mascarar_regioes_fixas(cv2.cvtColor(img_suavizada_rgb, cv2.COLOR_RGB2GRAY))

            circulos = detectar_bolhas_hough(
                img_processada_gray, dp=dp, min_dist=min_dist,
                param1=param1, param2=param2, min_radius=min_radius, max_radius=max_radius
            )
            circulos = remover_circulos_duplicados(circulos, fator=0.5) # Ajuste leve no fator de duplicação

            st.markdown("---")
            st.markdown("### 3. Resultados do Tratamento Inicial")

            colt1, colt2, colt3 = st.columns(3)
            with colt1:
                st.caption("Imagem original (RGB)")
                st.image(img_original_rgb, use_container_width=True)
            with colt2:
                st.caption("Imagem suavizada e com contraste (entrada da detecção)")
                st.image(img_suavizada_rgb, use_container_width=True)
            with colt3:
                # Imagem anotada com as bolhas medidas
                img_anotada_bgr = desenhar_bolhas(img_original_bgr, circulos, px_per_mm)
                img_anotada_rgb = cv2.cvtColor(img_anotada_bgr, cv2.COLOR_BGR2RGB)
                st.caption(f"Bolhas detectadas e medidas: {len(circulos)}")
                st.image(img_anotada_rgb, use_container_width=True)

            if not circulos:
                st.warning("Nenhuma bolha foi detectada com os parâmetros atuais. Tente reduzir o 'param2' ou ajustar os raios.")
                return

            # ... (restante do código para indicadores, gráficos e tabelas permanece igual) ...
            # DataFrame de medidas
            df = montar_dataframe_medidas(circulos, px_per_mm)
            # ... (restante do código) ...

        except Exception as e:
            st.error(f"Erro ao carregar/processar a imagem: {e}")
