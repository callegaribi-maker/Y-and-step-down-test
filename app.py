import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="Visualizador de Sinais", layout="wide")
st.title("📊 Visualizador de Sinais — Sincronização por L5")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def try_numeric(series):
    try:
        return pd.to_numeric(
            series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    except Exception:
        return pd.to_numeric(series, errors="coerce")


def load_file(uploaded_file):
    content = uploaded_file.read()
    uploaded_file.seek(0)
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    separators = [";", ",", "\t", r"\s+"]
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    io.BytesIO(content), sep=sep, engine="python",
                    encoding=enc, on_bad_lines="skip",
                )
                if df.shape[1] > 1:
                    for col in df.columns:
                        converted = try_numeric(df[col])
                        if converted.notna().sum() > len(df) * 0.5:
                            df[col] = converted
                    return df
            except Exception:
                continue
    return None


def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def l5_default(cols):
    for i, c in enumerate(cols):
        if "l5" in str(c).lower():
            return i
    return 0


def get_aligned_data(files_data, offsets):
    """
    Corta todos os sinais para a janela de sobreposição comum após aplicar os offsets.
    Retorna (dict{fname: df_cortado}, mensagem_info) ou (None, mensagem_erro).
    """
    if not files_data:
        return None, "Nenhum arquivo carregado."

    # Cada arquivo ocupa o intervalo [offset, offset + len(df)] no eixo global
    common_start = int(max(offsets.get(f, 0) for f in files_data))
    common_end   = int(min(offsets.get(f, 0) + len(df) for f, df in files_data.items()))

    if common_start >= common_end:
        return None, (
            f"Sem sobreposição após sincronização "
            f"(início comum: {common_start}, fim comum: {common_end}). "
            "Verifique os offsets."
        )

    aligned = {}
    for fname, df in files_data.items():
        shift     = offsets.get(fname, 0)
        idx_start = int(common_start - shift)
        idx_end   = int(common_end   - shift)
        aligned[fname] = df.iloc[idx_start:idx_end].reset_index(drop=True)

    n = common_end - common_start
    return aligned, f"Janela comum: **{n} amostras** ({n} pts compartilhados por todos os arquivos)."


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "files_data" not in st.session_state:
    st.session_state.files_data = {}
if "offsets" not in st.session_state:
    st.session_state.offsets = {}   # {fname: int}
if "show_l5_preview" not in st.session_state:
    st.session_state.show_l5_preview = False

# ──────────────────────────────────────────────
# Sidebar — 1. Upload
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Carregar Arquivos")
    uploaded = st.file_uploader(
        "Selecione até 5 arquivos (CSV ou TXT)",
        type=["csv", "txt"],
        accept_multiple_files=True,
    )
    if uploaded:
        loaded, errors = {}, []
        for f in uploaded:
            df = load_file(f)
            if df is not None:
                loaded[f.name] = df
            else:
                errors.append(f.name)
        if set(loaded.keys()) != set(st.session_state.files_data.keys()):
            st.session_state.files_data = loaded
            st.session_state.offsets = {}
        if errors:
            st.error(f"Não carregou: {', '.join(errors)}")
        st.success(f"{len(loaded)} arquivo(s) ✔")

files_data = st.session_state.files_data

if not files_data:
    st.info("👈 Carregue os arquivos na barra lateral para começar.")
    st.stop()

file_names = list(files_data.keys())

# ──────────────────────────────────────────────
# Sidebar — 2. Referência Kinem
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("2 · Referência (Kinem)")

    def default_idx(names, kw):
        for i, n in enumerate(names):
            if kw in n.lower():
                return i
        return 0

    kinem_ref = st.selectbox(
        "Arquivo de referência (offset = 0)",
        file_names,
        index=default_idx(file_names, "kinem"),
    )
    kinem_num = numeric_cols(files_data[kinem_ref])
    l5_kinem = st.selectbox(
        "Coluna L5 no Kinem", kinem_num, index=l5_default(kinem_num)
    )

# ──────────────────────────────────────────────
# Sidebar — 3. L5 por arquivo + offsets
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("3 · L5 por arquivo")

    other_files = [n for n in file_names if n != kinem_ref]
    l5_per_file = {}   # {fname: col or None}

    for fname in other_files:
        num = numeric_cols(files_data[fname])
        opcoes = ["— sem L5 / manual —"] + num
        idx = l5_default(num) + 1  # +1 por causa da opção manual
        escolha = st.selectbox(f"{fname[:35]}", opcoes, index=idx, key=f"l5_{fname}")
        l5_per_file[fname] = None if escolha.startswith("—") else escolha

    st.divider()

    if st.button("👁 Ver L5 de todos os arquivos"):
        st.session_state.show_l5_preview = True

    st.divider()

    janela = st.number_input(
        "Buscar pico nas primeiras N amostras",
        min_value=100, max_value=100000, value=3000, step=100,
        help="Use um valor que cubra apenas o início do sinal, onde ocorre o pulo."
    )

    if st.button("🔄 Calcular offsets por pico de L5", type="primary"):
        s_k = try_numeric(files_data[kinem_ref][l5_kinem]).abs()
        peak_k = int(s_k.iloc[:janela].idxmax())
        offsets = {kinem_ref: 0}
        msgs = []
        for fname, col in l5_per_file.items():
            if col is not None:
                s_f = try_numeric(files_data[fname][col]).abs()
                peak_f = int(s_f.iloc[:janela].idxmax())
                off = peak_k - peak_f
                offsets[fname] = off
                msgs.append(f"**{fname[:30]}**: {off} amostras (pico @ {peak_f})")
            else:
                offsets[fname] = st.session_state.offsets.get(fname, 0)
        st.session_state.offsets = offsets
        st.success(f"Kinem: pico @ {peak_k}\n\n" + "\n\n".join(msgs))

    # Offsets manuais para arquivos sem L5
    manual_files = [f for f in other_files if l5_per_file[f] is None]
    if manual_files:
        st.markdown("**Offsets manuais:**")
        for fname in manual_files:
            val = st.number_input(
                fname[:35],
                value=int(st.session_state.offsets.get(fname, 0)),
                step=1,
                key=f"manual_{fname}",
            )
            st.session_state.offsets[fname] = val

# ──────────────────────────────────────────────
# Preview L5 (sem sincronização — dados brutos)
# ──────────────────────────────────────────────
if st.session_state.show_l5_preview:
    st.subheader("👁 Preview L5 — encontre onde está o pulo")
    st.caption("Use este gráfico para decidir quantas amostras iniciais cobrem o pulo e ajuste o campo 'Buscar pico nas primeiras N amostras' na barra lateral.")

    l5_all = [(kinem_ref, l5_kinem)]
    for fname, col in l5_per_file.items():
        if col:
            l5_all.append((fname, col))

    n_prev = len(l5_all)
    fig_prev = make_subplots(
        rows=n_prev, cols=1, shared_xaxes=False,
        subplot_titles=[f"{fn} · {c}" for fn, c in l5_all],
        vertical_spacing=0.06,
    )
    for row, (fname, col) in enumerate(l5_all, start=1):
        y = try_numeric(files_data[fname][col])
        x = np.arange(len(y))
        fig_prev.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name=f"{fname} · {col}", showlegend=False),
            row=row, col=1,
        )
    fig_prev.update_layout(
        height=250 * n_prev, template="plotly_white",
        title="Sinais L5 — sem sincronização",
        hovermode="x unified",
    )
    st.plotly_chart(fig_prev, use_container_width=True)
    if st.button("✖ Fechar preview"):
        st.session_state.show_l5_preview = False
    st.divider()

# ──────────────────────────────────────────────
# Seleção de colunas por arquivo
# ──────────────────────────────────────────────
st.subheader("Seleção de colunas por arquivo")

col_selections = {}
grid = st.columns(min(len(files_data), 3))

for i, (fname, df) in enumerate(files_data.items()):
    num = numeric_cols(df)
    with grid[i % 3]:
        st.markdown(f"**{fname}**")
        off = st.session_state.offsets.get(fname, 0)
        st.caption(f"offset: {off} amostras")
        sel = st.multiselect(
            label=f"cols_{fname}",
            options=num,
            default=num[:4] if len(num) >= 4 else num,
            label_visibility="collapsed",
        )
        col_selections[fname] = sel

# ──────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────
st.divider()

plot_mode = st.radio(
    "Modo de plot",
    ["Um gráfico por coluna", "Todos no mesmo gráfico"],
    horizontal=True,
)

if st.button("📈 Plotar sinais", type="primary", use_container_width=True):

    # ── Alinhar e cortar para janela comum ──
    aligned_data, align_msg = get_aligned_data(files_data, st.session_state.offsets)

    if aligned_data is None:
        st.error(align_msg)
        st.stop()

    st.info(align_msg)

    # ── Montar traces usando dados alinhados ──
    traces = []
    for fname, df in aligned_data.items():
        selected = col_selections.get(fname, [])
        if not selected:
            continue
        x = np.arange(len(df))
        for col in selected:
            if col in df.columns:
                traces.append((fname, col, x, try_numeric(df[col])))

    if not traces:
        st.warning("Nenhuma coluna selecionada.")

    elif plot_mode == "Todos no mesmo gráfico":
        fig = go.Figure()
        for fname, col, x, y in traces:
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{fname} · {col}"))
        fig.update_layout(
            title="Sinais Sincronizados",
            xaxis_title="Amostra (janela comum)", yaxis_title="Valor",
            height=600, hovermode="x unified", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        n = len(traces)
        fig = make_subplots(
            rows=n, cols=1, shared_xaxes=True,
            subplot_titles=[f"{fn} · {c}" for fn, c, *_ in traces],
            vertical_spacing=0.03,
        )
        for row, (fname, col, x, y) in enumerate(traces, start=1):
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", name=f"{fname} · {col}"),
                row=row, col=1,
            )
        fig.update_layout(
            height=220 * n, hovermode="x unified",
            template="plotly_white", title="Sinais Sincronizados",
        )
        fig.update_xaxes(title_text="Amostra (janela comum)", row=n, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # ── Verificação: todos os L5 sobrepostos (dados alinhados) ──
    l5_cols = [(kinem_ref, l5_kinem)]
    for fname, col in l5_per_file.items():
        if col:
            l5_cols.append((fname, col))

    if len(l5_cols) > 1:
        with st.expander("🔍 Verificação — alinhamento dos L5"):
            fig3 = go.Figure()
            for fname, col in l5_cols:
                if fname in aligned_data and col in aligned_data[fname].columns:
                    x = np.arange(len(aligned_data[fname]))
                    fig3.add_trace(go.Scatter(
                        x=x,
                        y=try_numeric(aligned_data[fname][col]),
                        mode="lines",
                        name=f"{fname} · {col}",
                    ))
            fig3.update_layout(
                title="L5 — todos os arquivos sincronizados (janela comum)",
                xaxis_title="Amostra (janela comum)",
                hovermode="x unified",
                template="plotly_white",
                height=380,
            )
            st.plotly_chart(fig3, use_container_width=True)
