import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as sp_signal
from scipy import interpolate
import unicodedata
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
    """Retorna índice da primeira coluna que contém algum keyword (sem acentos, case-insensitive)."""
    normed_cols = [norm(c) for c in cols]
    for kw in keywords:
        kw_n = norm(kw)
        for i, cn in enumerate(normed_cols):
            if kw_n in cn:
                return i
    return 0


def norm(s):
    """Normaliza string: minúsculas + remove acentos."""
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s).lower())
        if unicodedata.category(c) != "Mn"
    )


def axis_label(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr):
    """
    Retorna o rótulo anatômico do eixo (ex: 'Vertical', 'ML', 'AP').
    Celular: X=Mediolateral, Y=Vertical, Z=Anteroposterior
    Kinem:   X=Mediolateral, Y=Anteroposterior, Z=Vertical
    """
    cn = norm(col)

    # Detecta o eixo: procura (x)/(y)/(z) ou termina com x/y/z
    axis = None
    for ax in ["x", "y", "z"]:
        if f"({ax})" in cn:
            axis = ax
            break
    if axis is None:
        for ax in ["z", "y", "x"]:          # z primeiro para não pegar "kx"
            if cn.rstrip().endswith(ax):
                axis = ax
                break

    if axis is None:
        return ""

    is_l5_phone   = fname in (l5_acc, l5_gyr)
    is_knee_phone = fname in (knee_acc, knee_gyr)
    is_kinem      = fname == kinem_ref

    if is_l5_phone:
        mapping = {"x": "ML", "y": "Vertical", "z": "AP"}
    elif is_knee_phone:
        mapping = {"x": "AP", "y": "Vertical", "z": "ML"}
    elif is_kinem:
        mapping = {"x": "ML", "y": "AP", "z": "Vertical"}
    else:
        return ""

    return mapping.get(axis, "")


def display_col_name(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr):
    """Nome original + rótulo anatômico entre parênteses."""
    lbl = axis_label(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr)
    return f"{col}  ({lbl})" if lbl else col


def classify_trace(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr):
    """Retorna 'l5', 'joelho' ou 'outro'."""
    if fname in (l5_acc, l5_gyr):
        return "l5"
    if fname in (knee_acc, knee_gyr):
        return "joelho"
    if fname == kinem_ref:
        cn = norm(col)
        if "l5" in cn or "l 5" in cn:
            return "l5"
        if any(k in cn for k in ["condilo", "joelho", "knee", "patela"]):
            return "joelho"
    return "outro"


def detect_time_axis(df):
    """
    Detecta coluna de tempo. Retorna (t_em_segundos, nome_coluna) ou (None, None).
    Reconhece: TempoMs (ms→s), Time / Tempo / t (já em segundos).
    """
    for col in df.columns:
        cl = str(col).lower().strip()
        if cl in ["tempoms", "tempo_ms", "time_ms", "timestamp_ms"]:
            return df[col].values.astype(float) / 1000.0, col
        if cl in ["time", "tempo", "t", "timestamp", "tempo (s)", "time (s)"]:
            return df[col].values.astype(float), col
    return None, None


def resample_to_regular(df, fs_target):
    """
    Reamostra df para grade regular em fs_target Hz usando o eixo de tempo real.
    Retorna (df_reamostrado, fs_original_detectada, descricao).
    """
    t, time_col = detect_time_axis(df)

    if t is None:
        # Sem coluna de tempo — não é possível reamostrar corretamente
        return df, None, "sem coluna de tempo (não reamostrado)"

    data_cols = [c for c in df.columns if c != time_col]
    t_norm    = t - t[0]          # normaliza para começar em 0
    duration  = t_norm[-1]
    fs_orig   = (len(t) - 1) / duration if duration > 0 else fs_target

    n_target = max(2, int(round(duration * fs_target)))
    t_target = np.linspace(0, duration, n_target)

    result = {}
    for col in data_cols:
        y = df[col].values
        if np.issubdtype(np.array(y).dtype, np.number):
            y = np.where(np.isnan(y.astype(float)), 0.0, y.astype(float))
            f_interp = interpolate.interp1d(
                t_norm, y, kind="linear",
                bounds_error=False, fill_value="extrapolate",
            )
            result[col] = f_interp(t_target)

    return pd.DataFrame(result), fs_orig, f"~{fs_orig:.0f} Hz → {fs_target} Hz"


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
    # sosfiltfilt é numericamente estável mesmo para cutoff muito baixo
    sos = sp_signal.butter(order, cutoff_hz / nyq, btype="low", output="sos")
    for col in numeric_cols(df):
        y = df[col].fillna(0).values
        filtered = sp_signal.sosfiltfilt(sos, y)
        result[col] = filtered
    return result


def get_aligned_data(files_data, offsets, peak_ref, ref_file=None):
    """
    Alinha todos os arquivos usando ref_file (ex: Kinem) como comprimento de referência.
    Arquivos mais curtos são preenchidos com NaN — aparecem como lacunas no gráfico.
    """
    common_start = int(max(offsets.get(f, 0) for f in files_data))

    # Usa ref_file para definir o fim da janela; senão, usa o mínimo comum
    if ref_file and ref_file in files_data:
        common_end = int(offsets.get(ref_file, 0) + len(files_data[ref_file]))
    else:
        common_end = int(min(offsets.get(f, 0) + len(df) for f, df in files_data.items()))

    if common_start >= common_end:
        return None, None, "Sem sobreposição após sincronização."

    n = common_end - common_start
    aligned = {}
    short_files = []

    for fname, df in files_data.items():
        s        = offsets.get(fname, 0)
        i_start  = int(common_start - s)          # índice de início no df original
        i_end    = int(common_end   - s)           # índice de fim   no df original
        a_start  = max(0, i_start)
        a_end    = min(len(df), i_end)

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        chunk    = df.iloc[a_start:a_end][num_cols].reset_index(drop=True)

        pad_before = a_start - i_start             # amostras em falta no início
        pad_after  = n - pad_before - len(chunk)   # amostras em falta no fim

        if pad_before > 0 or pad_after > 0:
            short_files.append(f"{fname} (faltam {max(0,pad_after)} amostras no fim)")
            rows = {}
            for col in num_cols:
                rows[col] = np.concatenate([
                    np.full(pad_before, np.nan),
                    chunk[col].values,
                    np.full(max(0, pad_after), np.nan),
                ])
            aligned[fname] = pd.DataFrame(rows)
        else:
            aligned[fname] = chunk

    peak_in_window = int(peak_ref - common_start)
    x_axis = np.arange(n) - peak_in_window
    info = f"Janela: **{n} amostras** ({n/100:.1f} s) | pico em **x = 0**"
    if short_files:
        info += f"  ⚠️ arquivos mais curtos que o Kinem: {', '.join(short_files)}"
    return aligned, x_axis, info


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
for k, v in [
    ("files_data", {}),
    ("proc_data",  {}),
    ("offsets",    {}),
    ("peak_ref",   0),
    ("target_fs",  100),
    ("fs_info",    {}),
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
            st.session_state.fs_info    = {}
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
    st.caption("As duas colunas são do mesmo arquivo — pico ocorre na mesma amostra.")
    st.caption("⚠️ No Kinem: Vertical = Z, AP = Y, ML = X. Selecione a coluna Z do L5 para sync.")
    l5_kinem_col = st.selectbox(
        "Coluna L5 vertical (referência sync)",
        kinem_num,
        index=col_default(kinem_num, ["l 5 a(z)", "l5 a(z)", "l5a(z)", "l 5 z", "l5_az", "l5"]),
    )
    knee_kinem_col = st.selectbox(
        "Coluna Joelho vertical (verificação)",
        kinem_num,
        index=col_default(kinem_num, [
            "condilo lateral esq. a(z)", "condilo lateral dir. a(z)",
            "condilo a(z)", "joelho a(z)", "knee a(z)",
            "condilo lateral esq.", "condilo", "côndilo", "joelho", "knee",
        ]),
    )

# ──────────────────────────────────────────────
# Sidebar — 3. Grupo L5
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

    l5_acc = st.selectbox("ACC L5", [NONE] + others,
                           index=best_match(others, ("acel", "l5"), ("acc", "l5")))
    l5_acc_col = None
    if l5_acc != NONE:
        num = numeric_cols(files_data[l5_acc])
        l5_acc_col = st.selectbox("Coluna Y do ACC L5", num,
                                   index=col_default(num, ["y"]), key="l5_acc_col")
    l5_gyr = st.selectbox("GYR L5  ← offset = ACC", [NONE] + others,
                           index=best_match(others, ("gyro", "l5"), ("gyr", "l5")))

# ──────────────────────────────────────────────
# Sidebar — 4. Grupo Joelho
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("4 · Grupo Joelho (celular)")
    knee_acc = st.selectbox("ACC Joelho", [NONE] + others,
                             index=best_match(others, ("acel", "joelho"), ("acc", "knee"), ("acel", "jo")))
    knee_acc_col = None
    if knee_acc != NONE:
        num = numeric_cols(files_data[knee_acc])
        knee_acc_col = st.selectbox("Coluna Y do ACC Joelho", num,
                                     index=col_default(num, ["y"]), key="knee_acc_col")
    knee_gyr = st.selectbox("GYR Joelho  ← offset = ACC", [NONE] + others,
                             index=best_match(others, ("gyro", "joelho"), ("gyr", "knee"), ("gyro", "jo")))

# ──────────────────────────────────────────────
# Sidebar — 5. Pré-processamento & Sync
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("5 · Pré-processamento & Sync")
    st.caption("A frequência de aquisição é detectada automaticamente da coluna de tempo de cada arquivo.")

    fs_target = st.number_input(
        "Frequência alvo após reamostragem (Hz)",
        min_value=1, max_value=10000, value=100, step=10,
        help="Todos os arquivos serão reamostrados para esta frequência comum.",
    )

    st.markdown("**Filtros opcionais**")
    do_detrend = st.checkbox("Detrend (remover tendência linear)", value=False)
    do_lowpass = st.checkbox("Filtro passa-baixa (Butterworth)", value=False)
    if do_lowpass:
        cutoff_hz  = st.number_input("Frequência de corte (Hz)",
                                      min_value=0.1, max_value=float(fs_target // 2),
                                      value=min(20.0, float(fs_target // 2 - 1)), step=0.5)
        filt_order = st.selectbox("Ordem do filtro", [2, 4, 6, 8], index=1)
    else:
        cutoff_hz, filt_order = 20.0, 4

    st.divider()

    if st.button("👁 Preview sinais brutos"):
        st.session_state.show_preview = not st.session_state.show_preview

    janela_seg = st.number_input(
        "Buscar pico nos primeiros X segundos",
        min_value=0.1, max_value=300.0, value=5.0, step=0.5,
        help="Use um valor pequeno que cubra só o início onde está o salto de sync.",
    )

    if st.button("⚙️ Pré-processar e Sincronizar", type="primary", use_container_width=True):
        with st.spinner("Detectando frequências e reamostando…"):
            proc     = {}
            fs_info  = {}
            msgs_pre = []

            proc_nofilter = {}   # reamostrado + detrend, SEM lowpass (para verificação)
            for fname, df in files_data.items():
                r, fs_orig, desc = resample_to_regular(df, fs_target)
                if do_detrend:
                    r = apply_detrend(r)
                proc_nofilter[fname] = r
                if do_lowpass:
                    r = apply_lowpass(r, fs_target, cutoff_hz, filt_order)
                proc[fname]    = r
                fs_info[fname] = fs_orig
                msgs_pre.append(f"**{fname[:35]}**: {desc}")

            st.session_state.proc_data        = proc
            st.session_state.proc_data_nofilter = proc_nofilter
            st.session_state.target_fs = fs_target
            st.session_state.fs_info   = fs_info

            # Sincronizar por pico
            janela_samp = int(janela_seg * fs_target)
            offsets = {kinem_ref: 0}
            msgs_sync = []

            s_k    = try_numeric(proc[kinem_ref][l5_kinem_col]).abs()
            peak_k = int(s_k.iloc[:janela_samp].idxmax())
            st.session_state.peak_ref = peak_k
            msgs_sync.append(f"**Kinem** — pico @ {peak_k} ({peak_k/fs_target:.2f} s)")

            if l5_acc != NONE and l5_acc_col and l5_acc_col in proc.get(l5_acc, pd.DataFrame()).columns:
                s = try_numeric(proc[l5_acc][l5_acc_col]).abs()
                p = int(s.iloc[:janela_samp].idxmax())
                off = peak_k - p
                offsets[l5_acc] = off
                msgs_sync.append(f"**L5 ACC** — pico @ {p} ({p/fs_target:.2f} s) → offset {off:+d}")
                if l5_gyr != NONE:
                    offsets[l5_gyr] = off
                    msgs_sync.append(f"**L5 GYR** — offset {off:+d} (= ACC)")

            if knee_acc != NONE and knee_acc_col and knee_acc_col in proc.get(knee_acc, pd.DataFrame()).columns:
                s = try_numeric(proc[knee_acc][knee_acc_col]).abs()
                p = int(s.iloc[:janela_samp].idxmax())
                off = peak_k - p
                offsets[knee_acc] = off
                msgs_sync.append(f"**Joelho ACC** — pico @ {p} ({p/fs_target:.2f} s) → offset {off:+d}")
                if knee_gyr != NONE:
                    offsets[knee_gyr] = off
                    msgs_sync.append(f"**Joelho GYR** — offset {off:+d} (= ACC)")

            for fname in file_names:
                if fname not in offsets:
                    offsets[fname] = 0

            st.session_state.offsets = offsets

            st.markdown("**Frequências detectadas:**")
            for m in msgs_pre:
                st.write(m)
            st.markdown("**Sincronização:**")
            for m in msgs_sync:
                st.write(m)

# ──────────────────────────────────────────────
# Preview bruto
# ──────────────────────────────────────────────
if st.session_state.show_preview:
    st.subheader("👁 Sinais brutos — sem pré-processamento")
    sync_cols = [(kinem_ref, l5_kinem_col)]
    if knee_kinem_col and knee_kinem_col != l5_kinem_col:
        sync_cols.append((kinem_ref, knee_kinem_col))
    if l5_acc != NONE and l5_acc_col:
        sync_cols.append((l5_acc, l5_acc_col))
    if knee_acc != NONE and knee_acc_col:
        sync_cols.append((knee_acc, knee_acc_col))

    pc1, pc2 = st.columns(2)
    with pc1:
        prev_t_start = st.number_input("Ver a partir de (s)", min_value=0.0, value=0.0, step=1.0, key="prev_start")
    with pc2:
        prev_t_end = st.number_input("Até (s)  — 0 = fim do sinal", min_value=0.0, value=0.0, step=1.0, key="prev_end")

    n_prev = len(sync_cols)
    fig_p = make_subplots(rows=n_prev, cols=1, shared_xaxes=False,
                           subplot_titles=[f"{fn} · {c}" for fn, c in sync_cols],
                           vertical_spacing=0.08)
    for row, (fname, col) in enumerate(sync_cols, start=1):
        t, tcol = detect_time_axis(files_data[fname])
        x = t - t[0] if t is not None else np.arange(len(files_data[fname]))
        y = try_numeric(files_data[fname][col])
        mask = x >= prev_t_start
        if prev_t_end > prev_t_start:
            mask &= x <= prev_t_end
        fig_p.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="lines", showlegend=False), row=row, col=1)
    fig_p.update_layout(height=280 * n_prev, template="plotly_white",
                         title="Colunas de sync — tempo original de cada arquivo",
                         hovermode="x unified")
    st.plotly_chart(fig_p, use_container_width=True)
    st.divider()

# ──────────────────────────────────────────────
# Seleção de colunas
# ──────────────────────────────────────────────
display_data = st.session_state.proc_data if st.session_state.proc_data else files_data
target_fs    = st.session_state.target_fs
fs_info      = st.session_state.fs_info

st.subheader("Seleção de colunas por arquivo")
col_selections = {}
grid = st.columns(min(len(display_data), 3))

for i, (fname, df) in enumerate(display_data.items()):
    num = numeric_cols(df)
    with grid[i % 3]:
        off  = st.session_state.offsets.get(fname, 0)
        grp  = (" 🔵 Kinem" if fname == kinem_ref
                else " 🟢 L5" if fname in (l5_acc, l5_gyr)
                else " 🟠 Joelho" if fname in (knee_acc, knee_gyr)
                else "")
        fs_orig = fs_info.get(fname)
        proc_label = (f"reamostrado de ~{fs_orig:.0f}→{target_fs} Hz"
                      if fs_orig and st.session_state.proc_data else "bruto")
        st.markdown(f"**{fname}**{grp}")
        st.caption(f"offset: {off:+d} | {proc_label}")
        sel = st.multiselect(
            label=f"cols_{fname}",
            options=num,
            default=num[:4] if len(num) >= 4 else num,
            label_visibility="collapsed",
        )
        col_selections[fname] = sel

# ──────────────────────────────────────────────
# Plot — configuração
# ──────────────────────────────────────────────
st.divider()

x_unit    = "Segundos"
plot_mode = "L5 | Joelho (lado a lado)"

vc1, vc2 = st.columns(2)
with vc1:
    view_start = st.number_input("Mostrar a partir de (s)", value=0.0, step=1.0, key="view_start",
                                  help="Ex: 2 para pular o pico do salto. 0 = início.")
with vc2:
    view_end = st.number_input("Mostrar até (s)  — 0 = fim", value=0.0, step=1.0, key="view_end")

x_label = ("Tempo (s)  —  0 = pico do salto" if x_unit == "Segundos" else "Amostra  —  0 = pico do salto")

# ── Colunas para check de qualidade ───────────────────────────
with st.expander("⚙️ Colunas para check de qualidade (1 por fonte)"):
    st.caption("Escolha exatamente qual coluna usar de cada fonte. Os 4 sinais serão plotados sobrepostos (z-score).")
    qk1, qk2 = st.columns(2)

    # ---- Kinem ----
    with qk1:
        qa_kinem_l5_col = st.selectbox(
            "🔵 Kinem — L5", kinem_num, key="qa_kl5",
            index=col_default(kinem_num, [
                "l 5 d(z)", "l5 d(z)", "l 5 d(y)", "l5 d(y)",
                "l 5 p(z)", "l5 p(z)", "l 5 v(z)", "l5 v(z)",
                "l 5 a(z)", "l5 a(z)", "l 5 z", "l5",
            ]),
        )
    with qk2:
        qa_kinem_knee_col = st.selectbox(
            "🔵 Kinem — Joelho", kinem_num, key="qa_kknee",
            index=col_default(kinem_num, [
                "condilo lateral esq. a(z)", "condilo lateral dir. a(z)",
                "condilo a(z)", "joelho a(z)",
                "condilo d(z)", "condilo d(y)", "condilo p(z)", "condilo p(y)",
                "condilo v(z)", "condilo lateral esq.", "condilo",
                "joelho a(z)", "joelho d", "joelho p",
            ]),
        )

    # ---- Celular L5 ----
    l5_acc_num  = numeric_cols(display_data.get(l5_acc,  pd.DataFrame())) if l5_acc  != NONE else []
    l5_gyr_num  = numeric_cols(display_data.get(l5_gyr,  pd.DataFrame())) if l5_gyr  != NONE else []
    knee_acc_num = numeric_cols(display_data.get(knee_acc, pd.DataFrame())) if knee_acc != NONE else []
    knee_gyr_num = numeric_cols(display_data.get(knee_gyr, pd.DataFrame())) if knee_gyr != NONE else []

    with qk1:
        qa_acc_l5_col = st.selectbox(
            "🟢 ACC — L5", l5_acc_num if l5_acc_num else ["—"],
            key="qa_accl5",
            index=col_default(l5_acc_num, ["z", "y", "x"]) if l5_acc_num else 0,
        ) if l5_acc_num else None
        qa_gyr_l5_col = st.selectbox(
            "🟢 GYR — L5", l5_gyr_num if l5_gyr_num else ["—"],
            key="qa_gyrl5",
            index=col_default(l5_gyr_num, ["z", "y", "x"]) if l5_gyr_num else 0,
        ) if l5_gyr_num else None
    with qk2:
        qa_acc_knee_col = st.selectbox(
            "🟠 ACC — Joelho", knee_acc_num if knee_acc_num else ["—"],
            key="qa_accknee",
            index=col_default(knee_acc_num, ["z", "y", "x"]) if knee_acc_num else 0,
        ) if knee_acc_num else None
        qa_gyr_knee_col = st.selectbox(
            "🟠 GYR — Joelho", knee_gyr_num if knee_gyr_num else ["—"],
            key="qa_gyrknee",
            index=col_default(knee_gyr_num, ["z", "y", "x"]) if knee_gyr_num else 0,
        ) if knee_gyr_num else None

# ── Checar qualidade dos dados ─────────────────────────────────
show_qa = st.checkbox("🔍 Checar qualidade dos dados", value=False)
if show_qa:
    if not st.session_state.proc_data:
        st.warning("Clique em **⚙️ Pré-processar e Sincronizar** antes.")
        st.stop()

    qa_aligned, qa_samp, _ = get_aligned_data(
        st.session_state.proc_data,
        st.session_state.offsets,
        st.session_state.peak_ref,
        ref_file=kinem_ref,
    )
    if qa_aligned is None:
        st.error("Sem sobreposição após sincronização.")
        st.stop()

    qa_x = qa_samp / st.session_state.target_fs if x_unit == "Segundos" else qa_samp
    qa_xmin = view_start if view_start != 0.0 else float(qa_x.min())
    qa_xmax = view_end   if view_end > view_start else float(qa_x.max())
    mask = (qa_x >= qa_xmin) & (qa_x <= qa_xmax)
    x_view = qa_x[mask]

    def get_entry(fname, col_name):
        """Retorna (std, label, y_view) para fname+coluna, ou None se indisponível."""
        df = qa_aligned.get(fname)
        if df is None or col_name is None or col_name not in df.columns:
            return None
        y = try_numeric(df[col_name]).values[mask].astype(float)
        if np.all(np.isnan(y)):
            return None
        dcol = display_col_name(fname, col_name, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr)
        return (float(np.nanstd(y)), f"{fname[:22]} · {dcol}", y)

    # Monta os sinais: Kinem + ACC + GYR para cada grupo
    e_kl5      = get_entry(kinem_ref, qa_kinem_l5_col)
    e_accl5    = get_entry(l5_acc   if l5_acc   != NONE else "", qa_acc_l5_col)
    e_gyrl5    = get_entry(l5_gyr   if l5_gyr   != NONE else "", qa_gyr_l5_col)
    e_kknee    = get_entry(kinem_ref, qa_kinem_knee_col)
    e_accknee  = get_entry(knee_acc if knee_acc != NONE else "", qa_acc_knee_col)
    e_gyrknee  = get_entry(knee_gyr if knee_gyr != NONE else "", qa_gyr_knee_col)

    l5_top     = [e for e in [e_kl5,   e_accl5,   e_gyrl5]   if e]
    joelho_top = [e for e in [e_kknee, e_accknee, e_gyrknee] if e]

    qa_c1, qa_c2 = st.columns(2)
    for col_out, group, title in [
        (qa_c1, l5_top,     "🟢 L5 — Kinem vs Celular"),
        (qa_c2, joelho_top, "🟠 Joelho — Kinem vs Celular"),
    ]:
        with col_out:
            st.markdown(f"#### {title}")
            if not group:
                st.info("Nenhum sinal classificado neste grupo.")
            else:
                fig_qa = go.Figure()
                for std_val, lbl, y_raw in group:
                    # z-score para comparar padrões temporais entre sensores
                    mn, sd = np.nanmean(y_raw), np.nanstd(y_raw)
                    y_norm = (y_raw - mn) / sd if sd > 0 else y_raw - mn
                    fig_qa.add_trace(go.Scatter(
                        x=x_view, y=y_norm, mode="lines",
                        name=f"{lbl}  (σ_orig={std_val:.3f})",
                    ))
                fig_qa.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
                fig_qa.update_layout(
                    xaxis=dict(title=x_label, range=[qa_xmin, qa_xmax]),
                    yaxis_title="z-score",
                    height=380, template="plotly_white", hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                    margin=dict(t=30, b=40),
                )
                st.plotly_chart(fig_qa, use_container_width=True)

if st.button("📈 Plotar sinais sincronizados", type="primary", use_container_width=True):

    if not st.session_state.proc_data:
        st.warning("Clique em **⚙️ Pré-processar e Sincronizar** antes de plotar.")
        st.stop()

    aligned_data, x_samp, align_msg = get_aligned_data(
        st.session_state.proc_data,
        st.session_state.offsets,
        st.session_state.peak_ref,
        ref_file=kinem_ref,
    )
    if aligned_data is None:
        st.error(align_msg)
        st.stop()

    st.info(align_msg)

    x_axis  = x_samp / target_fs if x_unit == "Segundos" else x_samp
    x_min_data, x_max_data = float(x_axis.min()), float(x_axis.max())
    # Aplica janela de visualização definida pelo usuário
    x_min = view_start if view_start != 0.0 else x_min_data
    x_max = view_end   if view_end   >  view_start else x_max_data

    traces = []
    for fname, df in aligned_data.items():
        for col in col_selections.get(fname, []):
            if col in df.columns:
                traces.append((fname, col, x_axis, try_numeric(df[col])))

    if not traces:
        st.warning("Nenhuma coluna selecionada.")
    else:
        # L5 | Joelho lado a lado
        l5_traces, knee_traces, other_traces = [], [], []
        for t in traces:
            cat = classify_trace(t[0], t[1], kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr)
            if cat == "l5":      l5_traces.append(t)
            elif cat == "joelho": knee_traces.append(t)
            else:                 other_traces.append(t)

        def render_col_charts(trace_list):
            for fname, col, x, y in trace_list:
                dcol = display_col_name(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr)
                fig_i = go.Figure()
                fig_i.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=1.5), showlegend=False))
                fig_i.add_vline(x=0, line_dash="dash", line_color="gray",
                                annotation_text="salto", annotation_position="top right")
                fig_i.update_layout(
                    title=dict(text=f"<b>{fname[:28]}</b> · {dcol}", font_size=12),
                    xaxis=dict(title=x_label, range=[x_min, x_max]),
                    yaxis_title="", height=230,
                    margin=dict(t=42, b=38, l=55, r=10),
                    hovermode="x", template="plotly_white",
                )
                st.plotly_chart(fig_i, use_container_width=True)

        col_l5, col_knee = st.columns(2)
        with col_l5:
            st.markdown("#### 🟢 L5")
            render_col_charts(l5_traces)
        with col_knee:
            st.markdown("#### 🟠 Joelho")
            render_col_charts(knee_traces)
        if other_traces:
            st.markdown("#### Outros sinais")
            render_col_charts(other_traces)

    # ── Verificação de alinhamento (usa dados SEM lowpass para ver o pico real) ──
    raw_aligned, raw_x_samp, _ = get_aligned_data(
        st.session_state.get("proc_data_nofilter", st.session_state.proc_data),
        st.session_state.offsets,
        st.session_state.peak_ref,
        ref_file=kinem_ref,
    )
    if raw_aligned is None:
        raw_aligned = aligned_data
        raw_x_samp  = x_samp
    raw_x_axis = raw_x_samp / target_fs   # sempre em segundos

    def render_verification(title, kinem_col, phone_file, phone_col, label_k, label_p):
        check = []
        df_k = raw_aligned.get(kinem_ref, pd.DataFrame())
        if kinem_col in df_k.columns:
            check.append((df_k, kinem_col, label_k))
        if phone_file != NONE and phone_col and phone_file in raw_aligned:
            df_p = raw_aligned[phone_file]
            if phone_col in df_p.columns:
                check.append((df_p, phone_col, label_p))
        if len(check) < 2:
            return
        with st.expander(f"🔍 Verificação — alinhamento {title}"):
            # normaliza pelo pico (máx abs) → ambas as linhas ficam em ±1
            colors_v = ["blue", "red"]
            series = []
            captions = []
            for df_s, col, lbl in check:
                s = try_numeric(df_s[col]).fillna(0).values.astype(float)
                peak = np.nanmax(np.abs(s))
                s_norm = s / peak if peak > 0 else s
                series.append((s_norm, lbl))
                captions.append(f"`{col}`")
            caption_txt = "  |  ".join(
                f"{'🔵' if i==0 else '🔴'} **{series[i][1]}**: {captions[i]}"
                for i in range(len(series))
            )
            st.caption(caption_txt + "  ·  normalizado pelo pico  ·  sem filtro passa-baixa")
            fig_v = go.Figure()
            for i, (s_norm, lbl) in enumerate(series):
                fig_v.add_trace(go.Scatter(
                    x=raw_x_axis, y=s_norm, mode="lines",
                    line=dict(color=colors_v[i], width=2),
                    name=lbl, opacity=0.85,
                ))
            if len(series) == 2:
                fig_v.add_trace(go.Scatter(
                    x=raw_x_axis, y=series[0][0] - series[1][0], mode="lines",
                    line=dict(color="gray", width=1, dash="dot"),
                    name="Diferença",
                ))
            fig_v.add_vline(x=0, line_dash="dash", line_color="black",
                            annotation_text="salto", annotation_position="top right")
            fig_v.update_layout(
                title=f"{title} — normalizado pelo pico (sem filtro)",
                xaxis=dict(title="Tempo (s) — 0 = pico do salto", range=[-5, 5]),
                yaxis=dict(title="Amplitude norm.", range=[-1.3, 1.3]),
                hovermode="x unified", template="plotly_white", height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            )
            st.plotly_chart(fig_v, use_container_width=True)

    render_verification("L5",    l5_kinem_col,   l5_acc,   l5_acc_col,   "Kinem L5",   "ACC L5")
    render_verification("Joelho", knee_kinem_col, knee_acc, knee_acc_col, "Kinem Joelho", "ACC Joelho")
