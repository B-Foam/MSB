import os
import streamlit as st


def app_em_manutencao() -> bool:
    return (
        st.secrets.get("APP_MAINTENANCE", False)
        or os.getenv("APP_MAINTENANCE", "false").lower() == "true"
    )


def mensagem_manutencao() -> str:
    return (
        st.secrets.get("APP_MAINTENANCE_MESSAGE")
        or os.getenv(
            "APP_MAINTENANCE_MESSAGE",
            "Aplicativo em manutenção. Tente novamente mais tarde."
        )
    )


def verificar_manutencao():
    if app_em_manutencao():
        st.set_page_config(
            page_title="Aplicativo em manutenção",
            page_icon="🛠️",
            layout="centered",
        )

        st.markdown(
            """
            <div style="
                margin-top: 80px;
                padding: 32px;
                border-radius: 18px;
                background: #ffffff;
                color: #0A2A66;
                text-align: center;
                box-shadow: 0 4px 18px rgba(0,0,0,0.15);
            ">
                <h1>🛠️ Aplicativo em manutenção</h1>
                <p style="font-size: 18px;">
                    O sistema está temporariamente indisponível para ajustes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(mensagem_manutencao())
        st.stop()
