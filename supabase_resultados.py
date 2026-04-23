import os
from typing import Any, Dict, List, Optional, Tuple

from supabase import create_client, Client
import streamlit as st


@st.cache_resource
def get_supabase_client() -> Client:
   url = (
    st.secrets.get("SUPABASE_URL")
    or st.secrets.get("supabase", {}).get("SUPABASE_URL")
    or os.getenv("SUPABASE_URL")
)

key = (
    st.secrets.get("SUPABASE_KEY")
    or st.secrets.get("supabase", {}).get("SUPABASE_KEY")
    or os.getenv("SUPABASE_KEY")
)

    if not url or not key:
        raise ValueError("SUPABASE_URL ou SUPABASE_KEY não configurados.")

    return create_client(url, key)


def salvar_resultado_teste_supabase(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Salva 1 resultado de teste na tabela public.resultados_granulometria.
    Retorna (registro_salvo, erro).
    """
    try:
        supabase = get_supabase_client()

        registro = {
            "tag_teste": payload.get("tag_teste"),
            "percentual_bolhas_maiores_500_um": payload.get("percentual_bolhas_maiores_500_um"),
            "quantidade_total_bolhas": payload.get("quantidade_total_bolhas"),
            "grafico_barras_resultados": payload.get("grafico_barras_resultados"),
            "tabela_bolhas": payload.get("tabela_bolhas"),
        }

        resp = (
            supabase.table("resultados_granulometria")
            .insert(registro)
            .execute()
        )

        data = getattr(resp, "data", None)
        if data and len(data) > 0:
            return data[0], None

        return None, "Supabase não retornou dados após o insert."

    except Exception as e:
        return None, str(e)


def listar_resultados_granulometria_supabase() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Lista os resultados gravados no Supabase.
    Retorna (lista, erro).
    """
    try:
        supabase = get_supabase_client()

        resp = (
            supabase.table("resultados_granulometria")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )

        data = getattr(resp, "data", None)
        if data is None:
            return [], None

        return data, None

    except Exception as e:
        return [], str(e)
