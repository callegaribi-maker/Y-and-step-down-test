import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="Visualizador de Sinais", layout="wide")
st.title("📊 Visualizador de Sinais — Y-Balance & Step-Down")

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
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]:
        for sep in [";", ",", "\t", r"\s+"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(content), sep=sep, engine="python",
                    encoding=enc, on_bad_lines="skip",
                )
                if df.shape[1] > 1:
                    for col in df.columns:
                        conv = try_numeric(df[col])
                        if conv.notna().sum() > len(df) * 0.5:
                            df[col] = conv
                    return df
            except Exception:
                continue
    return None


def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def col_default(cols, keywords):
    """Retorna o índice da primeira coluna cujo nome contém alguma keyword."""
    for kw in keywords:
        for i, c in enumerate(cols):
            if kw in str(c).lower():
                return i
    return 0


def get_aligned_data(files_data, offsets, peak_ref):
    """
    Corta todos os sinais para a janela comum e seta x=0 no pico do salto.
    Retorna (aligned_dict, x_axis, msg) ou (None, None, msg_erro).
    """
    common_start = int(max(offsets.get(f, 0) for f in files_data))
    common_end   = int(min(offsets.get(f, 0) + len(df) for f, df in files_data.items()))

    if common_start >= common_end:
        return None, None, "Sem sobreposição após sincronização. Verifique os offsets."

    aligned = {}
    for fname, df in files_data.items():
        s = offsets.get(fname, 0)
        aligned[fname] = df.iloc[int(common_start - s):int(common_end - s)].reset_index(drop=True)

    peak_in_window = int(peak_ref - common_start)
    x_axis = np.arange(common_end - common_start) - peak_in_window
    n = common_end - common_start
    return aligned, x_axis, f"Janela comum: **{n} amostras** | pico do salto em **x = 0**"


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
defaults = {
    "files_data": {},
    "offsets": {},
    "peak_ref": 0,
    "show_preview": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

NONE = "— nenhum —"

# ──────────────────────────────────────────────
# Sidebar — 1. Upload
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Carregar Arquivos")
    uploaded = st.file_uploader(
        "CSV ou TXT (até 5 arquivos)",
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
# Sidebar — 2. Kinem (referência)
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("2 · Kinem (referência)")
    kinem_idx = next((i for i, n in enumerate(file_names) if "kinem" in n.lower()), 0)
    kinem_ref = st.selectbox("Arquivo Kinem", file_names, index=kinem_idx)
    kinem_num = numeric_cols(files_data[kinem_ref])
    st.caption("As duas colunas são do mesmo arquivo — o pico do salto ocorre na mesma amostra para L5 e joelho.")
    l5_kinem_col = st.selectbox(
        "Coluna L5 vertical (referência de sync + verificação)",
        kinem_num,
        index=col_default(kinem_num, ["l5 a(z)", "l5a(z)", "l5_az", "l5"]),
    )
    knee_kinem_col = st.selectbox(
        "Coluna Joelho vertical (verificação)",
        kinem_num,
        index=col_default(kinem_num, ["joelho a(z)", "knee a(z)", "joelho", "knee"]),
    )

# ──────────────────────────────────────────────
# Sidebar — 3. Grupo L5 (celular)
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("3 · Grupo L5 (celular)")
    st.caption("ACC e GYR já estão sincronizados entre si — só precisa alinhar o ACC com o Kinem.")

    others = [n for n in file_names if n != kinem_ref]

    def best_match(names, *keywords):
        """Retorna índice (base 1 para incluir NONE) do melhor match."""
        for kw_set in keywords:
            for i, n in enumerate(names):
                if all(k in n.lower() for k in kw_set):
                    return i + 1   # +1 por causa do NONE na posição 0
        return 0

    l5_acc = st.selectbox(
        "ACC L5",
        [NONE] + others,
        index=best_match(others, ("acel", "l5"), ("acc", "l5")),
    )
    l5_acc_col = None
    if l5_acc != NONE:
        num = numeric_cols(files_data[l5_acc])
        l5_acc_col = st.selectbox(
            "Coluna Y do ACC L5",
            num,
            index=col_default(num, ["y"]),
            key="l5_acc_col",
        )

    l5_gyr = st.selectbox(
        "GYR L5  ← recebe mesmo offset do ACC",
        [NONE] + others,
        index=best_match(others, ("gyro", "l5"), ("gyr", "l5")),
    )

# ──────────────────────────────────────────────
# Sidebar — 4. Grupo Joelho (celular)
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("4 · Grupo Joelho (celular)")

    knee_acc = st.selectbox(
        "ACC Joelho",
        [NONE] + others,
        index=best_match(others, ("acel", "joelho"), ("acc", "knee"), ("acel", "jo")),
    )
    knee_acc_col = None
    if knee_acc != NONE:
        num = numeric_cols(files_data[knee_acc])
        knee_acc_col = st.selectbox(
            "Coluna Y do ACC Joelho",
            num,
            index=col_default(num, ["y"]),
            key="knee_acc_col",
        )

    knee_gyr = st.selectbox(
        "GYR Joelho  ← recebe mesmo offset do ACC",
        [NONE] + others,
        index=best_match(others, ("gyro", "joelho"), ("gyr", "knee"), ("gyro", "jo")),
    )

# ──────────────────────────────────────────────
# Sidebar — 5. Sincronizar
# ──────────────────────────────────────────────
with st.sidebar:
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("👁 Preview bruto"):
            st.session_state.show_preview = not st.session_state.show_preview
    with col_b:
        janela = st.number_input(
            "Janela (amostras)",
            min_value=100, max_value=500_000, value=5000, step=100,
            help="Número de amostras iniciais onde procurar o pico do salto.",
        )

    if st.button("🔄 Sincronizar por pico de salto", type="primary", use_container_width=True):
        offsets = {kinem_ref: 0}
        msgs = []

        # Pico de referência no Kinem
        s_k = try_numeric(files_data[kinem_ref][l5_kinem_col]).abs()
        peak_k = int(s_k.iloc[:janela].idxmax())
        st.session_state.peak_ref = peak_k
        msgs.append(f"**Kinem** — pico @ amostra {peak_k}")

        # Grupo L5
        if l5_acc != NONE and l5_acc_col:
            s = try_numeric(files_data[l5_acc][l5_acc_col]).abs()
            p = int(s.iloc[:janela].idxmax())
            off = peak_k - p
            offsets[l5_acc] = off
            msgs.append(f"**L5 ACC** — pico @ {p} → offset {off:+d}")
            if l5_gyr != NONE:
                offsets[l5_gyr] = off
                msgs.append(f"**L5 GYR** — offset {off:+d} (= ACC)")

        # Grupo Joelho
        if knee_acc != NONE and knee_acc_col:
            s = try_numeric(files_data[knee_acc][knee_acc_col]).abs()
            p = int(s.iloc[:janela].idxmax())
            off = peak_k - p
            offsets[knee_acc] = off
            msgs.append(f"**Joelho ACC** — pico @ {p} → offset {off:+d}")
            if knee_gyr != NONE:
                offsets[knee_gyr] = off
                msgs.append(f"**Joelho GYR** — offset {off:+d} (= ACC)")

        # Arquivos não atribuídos a nenhum grupo ficam com offset=0
        for fname in file_names:
            if fname not in offsets:
                offsets[fname] = 0
                msgs.append(f"**{fname[:30]}** — sem grupo, offset 0")

        st.session_state.offsets = offsets
        for m in msgs:
            st.write(m)

# ──────────────────────────────────────────────
# Preview bruto (sem sync)
# ──────────────────────────────────────────────
if st.session_state.show_preview:
    st.subheader("👁 Sinais brutos — sem sincronização")
    st.caption("Use para confirmar onde está o pico do salto e ajustar a janela de busca.")

    sync_cols = [(kinem_ref, l5_kinem_col)]
    if l5_acc != NONE and l5_acc_col:
        sync_cols.append((l5_acc, l5_acc_col))
    if knee_acc != NONE and knee_acc_col:
        sync_cols.append((knee_acc, knee_acc_col))

    n_prev = len(sync_cols)
    fig_prev = make_subplots(
        rows=n_prev, cols=1, shared_xaxes=False,
        subplot_titles=[f"{fn} · {c}" for fn, c in sync_cols],
        vertical_spacing=0.08,
    )
    for row, (fname, col) in enumerate(sync_cols, start=1):
        y = try_numeric(files_data[fname][col])
        fig_prev.add_trace(
            go.Scatter(x=np.arange(len(y)), y=y, mode="lines", showlegend=False),
            row=row, col=1,
        )
    fig_prev.update_layout(
        height=280 * n_prev, template="plotly_white",
        title="Colunas de sincronização — posição original",
        hovermode="x unified",
    )
    st.plotly_chart(fig_prev, use_container_width=True)
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
        off = st.session_state.offsets.get(fname, 0)
        grp = ""
        if fname == kinem_ref:
            grp = " 🔵 Kinem"
        elif fname in (l5_acc, l5_gyr):
            grp = " 🟢 L5"
        elif fname in (knee_acc, knee_gyr):
            grp = " 🟠 Joelho"
        st.markdown(f"**{fname}**{grp}")
        st.caption(f"offset: {off:+d} amostras")
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

if st.button("📈 Plotar sinais sincronizados", type="primary", use_container_width=True):

    aligned_data, x_axis, align_msg = get_aligned_data(
        files_data, st.session_state.offsets, st.session_state.peak_ref
    )

    if aligned_data is None:
        st.error(align_msg)
        st.stop()

    st.info(align_msg)

    traces = []
    for fname, df in aligned_data.items():
        selected = col_selections.get(fname, [])
        for col in selected:
            if col in df.columns:
                traces.append((fname, col, x_axis, try_numeric(df[col])))

    if not traces:
        st.warning("Nenhuma coluna selecionada.")

    elif plot_mode == "Todos no mesmo gráfico":
        fig = go.Figure()
        for fname, col, x, y in traces:
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{fname} · {col}"))
        fig.update_layout(
            title="Sinais Sincronizados",
            xaxis_title="Amostra (0 = pico do salto)",
            yaxis_title="Valor",
            height=600, hovermode="x unified", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
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
        for row in range(1, n + 1):
            fig.add_vline(x=0, line_dash="dash", line_color="gray", row=row, col=1)
        fig.update_layout(
            height=220 * n, hovermode="x unified",
            template="plotly_white", title="Sinais Sincronizados",
        )
        fig.update_xaxes(title_text="Amostra (0 = pico do salto)", row=n, col=1)
        st.plotly_chart(fig, use_container_width=True)

    # ── Verificação L5 ──────────────────────────────────────────
    l5_check = []
    if l5_kinem_col in aligned_data.get(kinem_ref, pd.DataFrame()).columns:
        l5_check.append((kinem_ref, l5_kinem_col, "Kinem L5"))
    if l5_acc != NONE and l5_acc_col and l5_acc in aligned_data:
        if l5_acc_col in aligned_data[l5_acc].columns:
            l5_check.append((l5_acc, l5_acc_col, "ACC L5"))

    if len(l5_check) > 1:
        with st.expander("🔍 Verificação — alinhamento L5"):
            fig_v = go.Figure()
            for fname, col, label in l5_check:
                y = try_numeric(aligned_data[fname][col])
                fig_v.add_trace(go.Scatter(x=x_axis, y=y, mode="lines", name=label))
            fig_v.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
            fig_v.update_layout(
                title="L5 — Kinem vs ACC (alinhados)",
                xaxis_title="Amostra (0 = pico)", hovermode="x unified",
                template="plotly_white", height=380,
            )
            st.plotly_chart(fig_v, use_container_width=True)

    # ── Verificação Joelho ──────────────────────────────────────
    knee_check = []
    if knee_kinem_col in aligned_data.get(kinem_ref, pd.DataFrame()).columns:
        knee_check.append((kinem_ref, knee_kinem_col, "Kinem Joelho"))
    if knee_acc != NONE and knee_acc_col and knee_acc in aligned_data:
        if knee_acc_col in aligned_data[knee_acc].columns:
            knee_check.append((knee_acc, knee_acc_col, "ACC Joelho"))

    if len(knee_check) > 1:
        with st.expander("🔍 Verificação — alinhamento Joelho"):
            fig_k = go.Figure()
            for fname, col, label in knee_check:
                y = try_numeric(aligned_data[fname][col])
                fig_k.add_trace(go.Scatter(x=x_axis, y=y, mode="lines", name=label))
            fig_k.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
            fig_k.update_layout(
                title="Joelho — Kinem vs ACC (alinhados)",
                xaxis_title="Amostra (0 = pico)", hovermode="x unified",
                template="plotly_white", height=380,
            )
            st.plotly_chart(fig_k, use_container_width=True)
