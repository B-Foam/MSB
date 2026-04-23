import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from supabase import Client, create_client


@st.cache_resource
def get_supabase_client() -> Client:
    secrets_supabase = st.secrets.get("supabase", {})

    url = (
        st.secrets.get("SUPABASE_URL")
        or secrets_supabase.get("SUPABASE_URL")
        or os.getenv("SUPABASE_URL")
    )

    key = (
        st.secrets.get("SUPABASE_KEY")
        or secrets_supabase.get("SUPABASE_KEY")
        or os.getenv("SUPABASE_KEY")
    )

    if not url or not key:
        raise ValueError("SUPABASE_URL ou SUPABASE_KEY não configurados.")

    return create_client(url, key)


def salvar_resultado_teste_supabase(
    payload: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Salva ou atualiza um teste na tabela resultados_granulometria usando tag_teste como chave única.
    Retorna:
        (registro_salvo, erro)
    """
    try:
        supabase = get_supabase_client()

        tag_teste = payload.get("tag_teste")
        if not tag_teste:
            return None, "tag_teste não informado."

        registro = {
            "tag_teste": tag_teste,
            "percentual_bolhas_maiores_500_um": payload.get("percentual_bolhas_maiores_500_um"),
            "quantidade_total_bolhas": payload.get("quantidade_total_bolhas"),
            "grafico_barras_resultados": payload.get("grafico_barras_resultados"),
            "tabela_bolhas": payload.get("tabela_bolhas"),
        }

        # upsert: se já existir o tag_teste, atualiza; se não existir, insere
        resp = (
            supabase.table("resultados_granulometria")
            .upsert(registro, on_conflict="tag_teste")
            .execute()
        )

        data = getattr(resp, "data", None)
        if data and len(data) > 0:
            return data[0], None

        # fallback: busca o registro salvo
        busca = (
            supabase.table("resultados_granulometria")
            .select("*")
            .eq("tag_teste", tag_teste)
            .limit(1)
            .execute()
        )

        data_busca = getattr(busca, "data", None)
        if data_busca and len(data_busca) > 0:
            return data_busca[0], None

        return None, "Supabase não retornou dados após o upsert."

    except Exception as e:
        return None, str(e)


def listar_resultados_granulometria_supabase() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Lista os resultados gravados no Supabase.
    Retorna:
        (lista_resultados, erro)
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
