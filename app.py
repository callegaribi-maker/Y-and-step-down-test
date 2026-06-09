import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as sp_signal
from scipy import interpolate
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
    for kw in keywords:
        for i, c in enumerate(cols):
            if kw in str(c).lower():
                return i
    return 0


def resample_df(df, fs_orig, fs_target):
    """Reamostra todas as colunas numéricas de fs_orig para fs_target."""
    if fs_orig == fs_target:
        return df
    n_orig = len(df)
    duration = n_orig / fs_orig
    n_target = int(round(duration * fs_target))
    t_orig   = np.linspace(0, duration, n_orig)
    t_target = np.linspace(0, duration, n_target)
    result = {}
    for col in df.columns:
        y = df[col].values
        if np.issubdtype(y.dtype, np.number):
            y = np.where(np.isnan(y), 0.0, y)
            f = interpolate.interp1d(t_orig, y, kind="linear",
                                     bounds_error=False, fill_value="extrapolate")
            result[col] = f(t_target)
        else:
            result[col] = [np.nan] * n_target
    return pd.DataFrame(result)


def apply_detrend(df):
    result = df.copy()
    for col in numeric_cols(df):
        result[col] = sp_signal.detrend(df[col].fillna(0).values)
    return result


def apply_lowpass(df, fs, cutoff_hz, order=4):
    result = df.copy()
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return result
    b, a = sp_signal.butter(order, cutoff_hz / nyq, btype="low")
    for col in numeric_cols(df):
        y = df[col].fillna(0).values
        result[col] = sp_signal.filtfilt(b, a, y)
    return result


def get_aligned_data(files_data, offsets, peak_ref):
    """Corta para janela comum e seta x=0 no pico."""
    common_start = int(max(offsets.get(f, 0) for f in files_data))
    common_end   = int(min(offsets.get(f, 0) + len(df) for f, df in files_data.items()))
    if common_start >= common_end:
        return None, None, "Sem sobreposição após sincronização."
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
for k, v in [
    ("files_data", {}),
    ("proc_data", {}),       # dados pós pré-processamento
    ("offsets", {}),
    ("peak_ref", 0),
    ("target_fs", 100),
    ("show_preview", False),
]:
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
            st.session_state.proc_data  = {}
            st.session_state.offsets    = {}
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
    st.caption("As duas colunas são do mesmo arquivo — pico do salto ocorre na mesma amostra.")
    l5_kinem_col = st.selectbox(
        "Coluna L5 vertical (referência sync)",
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
    st.caption("ACC e GYR já saem sincronizados entre si pelo celular.")

    others = [n for n in file_names if n != kinem_ref]

    def best_match(names, *kw_sets):
        for kws in kw_sets:
            for i, n in enumerate(names):
                if all(k in n.lower() for k in kws):
                    return i + 1
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
        "GYR L5  ← offset = ACC",
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
        "GYR Joelho  ← offset = ACC",
        [NONE] + others,
        index=best_match(others, ("gyro", "joelho"), ("gyr", "knee"), ("gyro", "jo")),
    )

# ──────────────────────────────────────────────
# Sidebar — 5. Pré-processamento + Sync
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("5 · Pré-processamento & Sync")

    st.markdown("**Frequências de aquisição (Hz)**")
    c1, c2 = st.columns(2)
    with c1:
        fs_kinem  = st.number_input("Kinem", min_value=1, max_value=10000, value=200, step=10)
    with c2:
        fs_celular = st.number_input("Celular", min_value=1, max_value=10000, value=100, step=10)

    fs_target = st.number_input(
        "Freq. alvo após reamostragem (Hz)",
        min_value=1, max_value=10000, value=100, step=10,
        help="Todos os sinais serão reamostrados para esta frequência antes do sync.",
    )

    st.markdown("**Filtros opcionais**")
    do_detrend = st.checkbox("Detrend (remover tendência linear)", value=False)
    do_lowpass = st.checkbox("Filtro passa-baixa (Butterworth)", value=False)
    if do_lowpass:
        cutoff_hz = st.number_input(
            "Frequência de corte (Hz)",
            min_value=0.1, max_value=float(fs_target // 2),
            value=min(20.0, float(fs_target // 2 - 1)),
            step=0.5,
        )
        filt_order = st.selectbox("Ordem do filtro", [2, 4, 6, 8], index=1)
    else:
        cutoff_hz  = 20.0
        filt_order = 4

    st.divider()

    if st.button("👁 Preview sinais brutos"):
        st.session_state.show_preview = not st.session_state.show_preview

    janela_seg = st.number_input(
        "Buscar pico nos primeiros X segundos",
        min_value=0.1, max_value=300.0, value=5.0, step=0.5,
        help="Após reamostrar, busca o pico neste intervalo inicial.",
    )

    if st.button("⚙️ Pré-processar e Sincronizar", type="primary", use_container_width=True):
        with st.spinner("Reamostando e processando…"):

            fs_map = {kinem_ref: fs_kinem}
            for fname in others:
                fs_map[fname] = fs_celular

            # 1. Reamostrar para fs_target
            proc = {}
            for fname, df in files_data.items():
                r = resample_df(df, fs_map[fname], fs_target)
                # 2. Detrend
                if do_detrend:
                    r = apply_detrend(r)
                # 3. Filtro passa-baixa
                if do_lowpass:
                    r = apply_lowpass(r, fs_target, cutoff_hz, filt_order)
                proc[fname] = r

            st.session_state.proc_data  = proc
            st.session_state.target_fs  = fs_target

            # 4. Sincronizar por pico (agora todos no mesmo fs_target)
            janela_samp = int(janela_seg * fs_target)
            offsets = {kinem_ref: 0}
            msgs = []

            s_k = try_numeric(proc[kinem_ref][l5_kinem_col]).abs()
            peak_k = int(s_k.iloc[:janela_samp].idxmax())
            st.session_state.peak_ref = peak_k
            msgs.append(f"**Kinem** — pico @ amostra {peak_k} ({peak_k/fs_target:.2f} s)")

            if l5_acc != NONE and l5_acc_col:
                s = try_numeric(proc[l5_acc][l5_acc_col]).abs()
                p = int(s.iloc[:janela_samp].idxmax())
                off = peak_k - p
                offsets[l5_acc] = off
                msgs.append(f"**L5 ACC** — pico @ {p} ({p/fs_target:.2f} s) → offset {off:+d}")
                if l5_gyr != NONE:
                    offsets[l5_gyr] = off
                    msgs.append(f"**L5 GYR** — offset {off:+d} (= ACC)")

            if knee_acc != NONE and knee_acc_col:
                s = try_numeric(proc[knee_acc][knee_acc_col]).abs()
                p = int(s.iloc[:janela_samp].idxmax())
                off = peak_k - p
                offsets[knee_acc] = off
                msgs.append(f"**Joelho ACC** — pico @ {p} ({p/fs_target:.2f} s) → offset {off:+d}")
                if knee_gyr != NONE:
                    offsets[knee_gyr] = off
                    msgs.append(f"**Joelho GYR** — offset {off:+d} (= ACC)")

            for fname in file_names:
                if fname not in offsets:
                    offsets[fname] = 0

            st.session_state.offsets = offsets
            for m in msgs:
                st.write(m)

# ──────────────────────────────────────────────
# Preview bruto
# ──────────────────────────────────────────────
if st.session_state.show_preview:
    st.subheader("👁 Sinais brutos — sem pré-processamento")
    sync_cols = [(kinem_ref, l5_kinem_col)]
    if l5_acc != NONE and l5_acc_col:
        sync_cols.append((l5_acc, l5_acc_col))
    if knee_acc != NONE and knee_acc_col:
        sync_cols.append((knee_acc, knee_acc_col))

    n_prev = len(sync_cols)
    fig_p = make_subplots(
        rows=n_prev, cols=1, shared_xaxes=False,
        subplot_titles=[f"{fn} · {c}" for fn, c in sync_cols],
        vertical_spacing=0.08,
    )
    for row, (fname, col) in enumerate(sync_cols, start=1):
        y = try_numeric(files_data[fname][col])
        fig_p.add_trace(
            go.Scatter(x=np.arange(len(y)), y=y, mode="lines", showlegend=False),
            row=row, col=1,
        )
    fig_p.update_layout(
        height=280 * n_prev, template="plotly_white",
        title="Colunas de sync — amostras originais (frequências diferentes!)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_p, use_container_width=True)
    st.divider()

# ──────────────────────────────────────────────
# Seleção de colunas
# ──────────────────────────────────────────────
# Usa proc_data se disponível, senão files_data
display_data = st.session_state.proc_data if st.session_state.proc_data else files_data
target_fs    = st.session_state.target_fs

st.subheader("Seleção de colunas por arquivo")
col_selections = {}
grid = st.columns(min(len(display_data), 3))

for i, (fname, df) in enumerate(display_data.items()):
    num = numeric_cols(df)
    with grid[i % 3]:
        off = st.session_state.offsets.get(fname, 0)
        grp = ""
        if fname == kinem_ref:          grp = " 🔵 Kinem"
        elif fname in (l5_acc, l5_gyr): grp = " 🟢 L5"
        elif fname in (knee_acc, knee_gyr): grp = " 🟠 Joelho"
        st.markdown(f"**{fname}**{grp}")
        proc_label = "pré-proc ✔" if st.session_state.proc_data else "bruto"
        st.caption(f"offset: {off:+d} amostras | {proc_label} @ {target_fs} Hz")
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

x_unit = st.radio("Eixo x", ["Amostras", "Segundos"], horizontal=True)

if st.button("📈 Plotar sinais sincronizados", type="primary", use_container_width=True):

    if not st.session_state.proc_data:
        st.warning("Clique em **⚙️ Pré-processar e Sincronizar** antes de plotar.")
        st.stop()

    aligned_data, x_samp, align_msg = get_aligned_data(
        st.session_state.proc_data,
        st.session_state.offsets,
        st.session_state.peak_ref,
    )
    if aligned_data is None:
        st.error(align_msg)
        st.stop()

    st.info(align_msg)

    x_axis  = x_samp / target_fs if x_unit == "Segundos" else x_samp
    x_label = "Tempo (s) — 0 = pico do salto" if x_unit == "Segundos" else "Amostra (0 = pico do salto)"

    traces = []
    for fname, df in aligned_data.items():
        for col in col_selections.get(fname, []):
            if col in df.columns:
                traces.append((fname, col, x_axis, try_numeric(df[col])))

    if not traces:
        st.warning("Nenhuma coluna selecionada.")

    elif plot_mode == "Todos no mesmo gráfico":
        fig = go.Figure()
        for fname, col, x, y in traces:
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{fname} · {col}"))
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
        fig.update_layout(
            title="Sinais Sincronizados",
            xaxis_title=x_label, yaxis_title="Valor",
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
        for row in range(1, n + 1):
            fig.add_vline(x=0, line_dash="dash", line_color="gray", row=row, col=1)
        fig.update_layout(
            height=220 * n, hovermode="x unified",
            template="plotly_white", title="Sinais Sincronizados",
        )
        fig.update_xaxes(title_text=x_label, row=n, col=1)
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
                fig_v.add_trace(go.Scatter(
                    x=x_axis, y=try_numeric(aligned_data[fname][col]),
                    mode="lines", name=label,
                ))
            fig_v.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
            fig_v.update_layout(
                title="L5 — Kinem vs ACC (alinhados, mesma fs)",
                xaxis_title=x_label, hovermode="x unified",
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
                fig_k.add_trace(go.Scatter(
                    x=x_axis, y=try_numeric(aligned_data[fname][col]),
                    mode="lines", name=label,
                ))
            fig_k.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
            fig_k.update_layout(
                title="Joelho — Kinem vs ACC (alinhados, mesma fs)",
                xaxis_title=x_label, hovermode="x unified",
                template="plotly_white", height=380,
            )
            st.plotly_chart(fig_k, use_container_width=True)
