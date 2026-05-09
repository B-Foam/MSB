import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def imagem_para_base64(caminho: Path) -> str:
    """Converte imagem local em base64 para usar dentro do HTML."""
    with open(caminho, "rb") as arquivo:
        return base64.b64encode(arquivo.read()).decode("utf-8")


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

    imagens = []
    arquivos_nao_encontrados = []

    for pagina in paginas:
        caminho = revista_dir / pagina["arquivo"]

        if caminho.exists():
            imagens.append(
                {
                    "titulo": pagina["titulo"],
                    "arquivo": pagina["arquivo"],
                    "src": f"data:image/png;base64,{imagem_para_base64(caminho)}",
                }
            )
        else:
            arquivos_nao_encontrados.append(str(caminho))

    if arquivos_nao_encontrados:
        st.error("Algumas imagens da revista não foram encontradas:")
        for arquivo in arquivos_nao_encontrados:
            st.write(f"- {arquivo}")
        st.warning("Verifique se os nomes dos arquivos estão iguais aos nomes salvos no GitHub.")
        return

    html_paginas = str(imagens).replace("</", "<\\/")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8" />
        <style>
            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, Helvetica, sans-serif;
                background: linear-gradient(135deg, #061b33, #0b2d55, #eaf4fb);
                overflow: hidden;
            }}

            .app {{
                width: 100%;
                height: 100vh;
                min-height: 820px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 18px;
                color: white;
            }}

            .topbar {{
                width: min(1180px, 100%);
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                margin-bottom: 12px;
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(255,255,255,0.20);
                border-radius: 18px;
                padding: 12px 16px;
                backdrop-filter: blur(10px);
            }}

            .title {{
                display: flex;
                flex-direction: column;
                gap: 2px;
            }}

            .title strong {{
                font-size: 17px;
                letter-spacing: 0.3px;
            }}

            .title span {{
                font-size: 12px;
                opacity: 0.85;
            }}

            .controls {{
                display: flex;
                align-items: center;
                gap: 8px;
                flex-wrap: wrap;
                justify-content: flex-end;
            }}

            button {{
                border: none;
                border-radius: 12px;
                padding: 9px 12px;
                font-weight: 700;
                cursor: pointer;
                color: #07305a;
                background: #ffffff;
                box-shadow: 0 4px 14px rgba(0,0,0,0.12);
                transition: transform 0.15s ease, background 0.15s ease;
            }}

            button:hover {{
                transform: translateY(-1px);
                background: #dff2ff;
            }}

            .counter {{
                font-size: 13px;
                padding: 8px 12px;
                border-radius: 12px;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.20);
                min-width: 130px;
                text-align: center;
            }}

            .viewer-wrap {{
                width: min(1180px, 100%);
                height: 76vh;
                min-height: 640px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: auto;
                background: rgba(255,255,255,0.94);
                border-radius: 24px;
                border: 1px solid rgba(255,255,255,0.35);
                box-shadow: 0 20px 55px rgba(0,0,0,0.30);
                position: relative;
            }}

            .viewer-wrap:fullscreen {{
                width: 100vw;
                height: 100vh;
                min-height: 100vh;
                border-radius: 0;
                background: #061b33;
                padding: 18px;
            }}

            .page-stage {{
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                transition: transform 0.25s ease;
            }}

            .page {{
                max-height: 72vh;
                max-width: 100%;
                border-radius: 12px;
                box-shadow: 0 14px 35px rgba(0,0,0,0.28);
                user-select: none;
                -webkit-user-drag: none;
                cursor: pointer;
                background: white;
                transition: opacity 0.18s ease, transform 0.18s ease;
            }}

            .page.changing {{
                opacity: 0.35;
                transform: scale(0.985);
            }}

            .hint {{
                width: min(1180px, 100%);
                margin-top: 10px;
                text-align: center;
                color: rgba(255,255,255,0.88);
                font-size: 13px;
            }}

            .arrow-zone {{
                position: absolute;
                top: 0;
                height: 100%;
                width: 25%;
                z-index: 5;
            }}

            .arrow-zone.left {{
                left: 0;
                cursor: w-resize;
            }}

            .arrow-zone.right {{
                right: 0;
                cursor: e-resize;
            }}

            @media (max-width: 768px) {{
                .app {{
                    padding: 10px;
                    min-height: 720px;
                }}

                .topbar {{
                    flex-direction: column;
                    align-items: stretch;
                }}

                .controls {{
                    justify-content: center;
                }}

                .viewer-wrap {{
                    height: 70vh;
                    min-height: 520px;
                }}

                .page {{
                    max-height: 66vh;
                }}

                button {{
                    padding: 8px 10px;
                    font-size: 12px;
                }}
            }}
        </style>
    </head>

    <body>
        <div class="app">
            <div class="topbar">
                <div class="title">
                    <strong id="pageTitle">Revista Técnica MSB</strong>
                    <span>Materiais para dispositivos médicos</span>
                </div>

                <div class="controls">
                    <button onclick="goFirst()">⏮ Início</button>
                    <button onclick="prevPage()">⬅ Anterior</button>
                    <div class="counter" id="counter">Página 1 de 1</div>
                    <button onclick="nextPage()">Próxima ➡</button>
                    <button onclick="goLast()">Fim ⏭</button>
                    <button onclick="zoomOut()">− Zoom</button>
                    <button onclick="zoomIn()">+ Zoom</button>
                    <button onclick="resetZoom()">100%</button>
                    <button onclick="toggleFullscreen()">⛶ Tela cheia</button>
                </div>
            </div>

            <div class="viewer-wrap" id="viewer">
                <div class="arrow-zone left" onclick="prevPage()"></div>
                <div class="arrow-zone right" onclick="nextPage()"></div>

                <div class="page-stage" id="stage">
                    <img id="pageImage" class="page" src="" alt="Página da revista MSB" onclick="nextPage()" />
                </div>
            </div>

            <div class="hint">
                Use o mouse: clique no lado direito para avançar, lado esquerdo para voltar, scroll para passar páginas,
                setas do teclado para navegar e botão de tela cheia para ampliar.
            </div>
        </div>

        <script>
            const pages = {html_paginas};

            let currentPage = 0;
            let zoom = 1;
            let lastWheelTime = 0;

            const pageImage = document.getElementById("pageImage");
            const pageTitle = document.getElementById("pageTitle");
            const counter = document.getElementById("counter");
            const stage = document.getElementById("stage");
            const viewer = document.getElementById("viewer");

            function renderPage() {{
                pageImage.classList.add("changing");

                setTimeout(() => {{
                    const page = pages[currentPage];
                    pageImage.src = page.src;
                    pageTitle.innerText = page.titulo;
                    counter.innerText = `Página ${{currentPage + 1}} de ${{pages.length}}`;
                    stage.style.transform = `scale(${{zoom}})`;
                    pageImage.classList.remove("changing");
                }}, 120);
            }}

            function nextPage() {{
                if (currentPage < pages.length - 1) {{
                    currentPage += 1;
                    renderPage();
                }}
            }}

            function prevPage() {{
                if (currentPage > 0) {{
                    currentPage -= 1;
                    renderPage();
                }}
            }}

            function goFirst() {{
                currentPage = 0;
                renderPage();
            }}

            function goLast() {{
                currentPage = pages.length - 1;
                renderPage();
            }}

            function zoomIn() {{
                zoom = Math.min(zoom + 0.15, 2.5);
                stage.style.transform = `scale(${{zoom}})`;
            }}

            function zoomOut() {{
                zoom = Math.max(zoom - 0.15, 0.5);
                stage.style.transform = `scale(${{zoom}})`;
            }}

            function resetZoom() {{
                zoom = 1;
                stage.style.transform = `scale(${{zoom}})`;
            }}

            function toggleFullscreen() {{
                if (!document.fullscreenElement) {{
                    viewer.requestFullscreen().catch(() => {{
                        alert("O navegador bloqueou a tela cheia. Tente clicar novamente no botão.");
                    }});
                }} else {{
                    document.exitFullscreen();
                }}
            }}

            document.addEventListener("keydown", function(event) {{
                if (event.key === "ArrowRight") {{
                    nextPage();
                }}

                if (event.key === "ArrowLeft") {{
                    prevPage();
                }}

                if (event.key === "+" || event.key === "=") {{
                    zoomIn();
                }}

                if (event.key === "-") {{
                    zoomOut();
                }}

                if (event.key === "Escape") {{
                    resetZoom();
                }}
            }});

            viewer.addEventListener("wheel", function(event) {{
                event.preventDefault();

                const now = Date.now();

                if (now - lastWheelTime < 450) {{
                    return;
                }}

                lastWheelTime = now;

                if (event.deltaY > 0) {{
                    nextPage();
                }} else {{
                    prevPage();
                }}
            }}, {{ passive: false }});

            let startX = 0;

            viewer.addEventListener("mousedown", function(event) {{
                startX = event.clientX;
            }});

            viewer.addEventListener("mouseup", function(event) {{
                const diff = event.clientX - startX;

                if (Math.abs(diff) > 60) {{
                    if (diff < 0) {{
                        nextPage();
                    }} else {{
                        prevPage();
                    }}
                }}
            }});

            renderPage();
        </script>
    </body>
    </html>
    """

    components.html(html, height=920, scrolling=False)


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
