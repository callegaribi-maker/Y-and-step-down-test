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


def is_xyz_col(col):
    """True se a coluna for eixo X, Y ou Z (exclui abs, magnitude, length)."""
    cn = norm(col).lower()
    for ex in ("abs", "magnitude", "length", "norma", "mag", "len", "norm"):
        if ex in cn:
            return False
    import re
    return bool(re.search(r'(?:^|[_\s\(])([xyz])(?:[_\s\)]|$)', cn))


def kinem_cols_for_body(df, *body_keywords):
    """
    Retorna colunas do Kinem para uma região anatômica.
    Inclui colunas cujo nome:
      - contém algum body_keyword
      - termina em X, Y ou Z  (ex: L5X, L5 X)
      - OU contém (X), (Y) ou (Z)  (ex: L5 v(Z), L5 a(Y))
    Exclui: abs, length, #2D e colunas de comprimento l(Z).
    """
    import re
    result = []
    for col in df.columns:
        cn = norm(col).lower().strip()
        if not any(kw in cn for kw in body_keywords):
            continue
        # exclui comprimento, valores absolutos e métricas 2D
        if "abs" in cn or "length" in cn or "#2d" in cn or re.search(r'\bl\(', cn):
            continue
        # inclui se tem eixo com parênteses ou se termina em x/y/z
        has_paren_axis  = any(f"({ax})" in cn for ax in ("x", "y", "z"))
        has_suffix_axis = bool(re.search(r'[xyz]$', cn))
        if has_paren_axis or has_suffix_axis:
            result.append(col)
    return result


def build_export_sheet(aligned, kinem_ref, acc_file, gyr_file,
                       kinem_keywords, t, fs, NONE="— nenhum —"):
    """Monta DataFrame de uma aba do Excel (L5 ou Joelho)."""
    dfs = [pd.DataFrame({"Tempo (s)": t})]

    # Kinem
    kdf = aligned.get(kinem_ref, pd.DataFrame())
    k_cols = kinem_cols_for_body(kdf, *kinem_keywords)
    if k_cols:
        dfs.append(kdf[k_cols].reset_index(drop=True))

    # Phone ACC
    if acc_file and acc_file != NONE and acc_file in aligned:
        adf  = aligned[acc_file]
        cols = [c for c in adf.columns if is_xyz_col(c)]
        if cols:
            dfs.append(adf[cols].add_prefix("ACC_").reset_index(drop=True))

    # Phone GYR
    if gyr_file and gyr_file != NONE and gyr_file in aligned:
        gdf  = aligned[gyr_file]
        cols = [c for c in gdf.columns if is_xyz_col(c)]
        if cols:
            dfs.append(gdf[cols].add_prefix("GYR_").reset_index(drop=True))

    result = pd.concat(dfs, axis=1)
    result = result.iloc[:len(t)]
    return result


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
        return df, None, "sem coluna de tempo (não reamostrado)"

    data_cols = [c for c in df.columns if c != time_col]
    t_norm    = t - t[0]
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


def _impact_envelope(v, fs=100.0):
    v = np.asarray(v, dtype=float)
    if len(v) < 12:
        return np.abs(v - np.mean(v))
    nyq    = fs / 2.0
    cutoff = min(1.0, nyq * 0.95)
    sos    = sp_signal.butter(2, cutoff / nyq, btype="high", output="sos")
    return np.abs(sp_signal.sosfiltfilt(sos, v))


def find_highest_peak(series, search_end, fs=100.0):
    """Pico de maior amplitude no envelope highpass dos primeiros search_end samples."""
    raw = try_numeric(series).fillna(0).values[:search_end].astype(float)
    if len(raw) == 0:
        return 0
    vals    = _impact_envelope(raw, fs)
    max_val = vals.max()
    if max_val == 0:
        return int(np.argmax(vals))
    peaks, _ = sp_signal.find_peaks(vals, prominence=max_val * 0.30)
    if len(peaks) == 0:
        return int(np.argmax(vals))
    return int(peaks[np.argmax(vals[peaks])])


def _local_corr(kinem_vals, phone_vals, kinem_peak, phone_peak, fs):
    win = int(fs)
    k_v = kinem_vals
    p_v = phone_vals
    ks  = max(0, kinem_peak - win);  ke = min(len(k_v), kinem_peak + win)
    ps  = max(0, phone_peak  - win);  pe = min(len(p_v), phone_peak  + win)
    k_seg = _impact_envelope(k_v[ks:ke], fs)
    p_seg = _impact_envelope(p_v[ps:pe], fs)
    n = min(len(k_seg), len(p_seg))
    if n < 4:
        return 0.0
    k_seg, p_seg = k_seg[:n], p_seg[:n]
    if k_seg.std() == 0 or p_seg.std() == 0:
        return 0.0
    return float(abs(np.corrcoef(k_seg, p_seg)[0, 1]))


def find_sync_xcorr(kinem_ser, phone_ser, kinem_peak, search_end, fs):
    k_vals = try_numeric(kinem_ser).fillna(0).values.astype(float)
    p_vals = try_numeric(phone_ser).fillna(0).values[:search_end].astype(float)

    p_simple = find_highest_peak(pd.Series(p_vals), len(p_vals), fs)

    half_tpl = int(2 * fs)
    k_start  = max(0, kinem_peak - half_tpl)
    k_end    = min(len(k_vals), kinem_peak + half_tpl)
    k_seg    = k_vals[k_start:k_end]

    p_xcorr = None
    if len(p_vals) >= len(k_seg) + 1 and len(k_seg) >= 4:
        ref_env   = _impact_envelope(k_seg,  fs)
        phone_env = _impact_envelope(p_vals, fs)
        corr      = np.correlate(phone_env, ref_env, mode="valid")
        lag       = int(np.argmax(corr))
        candidate = lag + (kinem_peak - k_start)
        if 0 <= candidate < search_end:
            p_xcorr = candidate

    if p_xcorr is None:
        return p_simple

    c_simple = _local_corr(k_vals, p_vals, kinem_peak, p_simple, fs)
    c_xcorr  = _local_corr(k_vals, p_vals, kinem_peak, p_xcorr,  fs)

    return p_xcorr if c_xcorr > c_simple else p_simple


def apply_lowpass(df, fs, cutoff_hz, order=4):
    result = df.copy()
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return result
    sos = sp_signal.butter(order, cutoff_hz / nyq, btype="low", output="sos")
    for col in numeric_cols(df):
        y = df[col].fillna(0).values
        filtered = sp_signal.sosfiltfilt(sos, y)
        result[col] = filtered
    return result


def get_aligned_data(files_data, offsets, peak_ref, ref_file=None):
    """
    Alinha todos os arquivos usando ref_file como comprimento de referência.
    Arquivos mais curtos são preenchidos com NaN.
    """
    common_start = int(max(offsets.get(f, 0) for f in files_data))

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
        i_start  = int(common_start - s)
        i_end    = int(common_end   - s)
        a_start  = max(0, i_start)
        a_end    = min(len(df), i_end)

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        chunk    = df.iloc[a_start:a_end][num_cols].reset_index(drop=True)

        pad_before = a_start - i_start
        pad_after  = n - pad_before - len(chunk)

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
    ("files_data",          {}),
    ("raw_synced",          {}),
    ("proc_data",           {}),
    ("proc_data_nofilter",  {}),
    ("offsets",             {}),
    ("peak_ref",            None),
    ("target_fs",           100),
    ("fs_info",             {}),
    ("show_preview",        False),
    ("synced",              False),
    ("synced_l5_col",       None),
    ("synced_knee_col",     None),
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
        "Coluna Joelho vertical (referência sync)",
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
# Sidebar — Configurações avançadas de sincronização
# ──────────────────────────────────────────────
with st.sidebar:
    with st.expander("⚙️ Configurações avançadas de sincronização", expanded=False):
        fs_target = st.number_input(
            "Frequência alvo após reamostragem (Hz)",
            min_value=1, max_value=10000, value=100, step=10,
            help="Todos os arquivos serão reamostrados para esta frequência comum.",
        )

# ──────────────────────────────────────────────
# Botões: Preview + Sincronizar
# ──────────────────────────────────────────────
btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])
with btn_col1:
    if st.button("👁 Preview sinais brutos", use_container_width=True):
        st.session_state.show_preview = not st.session_state.show_preview
with btn_col2:
    janela_seg = st.number_input(
        "Pico nos primeiros (s)",
        min_value=0.1, max_value=300.0, value=16.0, step=0.5,
        help="Janela de busca do pico de sincronização.",
    )
with btn_col3:
    if st.button("🔗 Sincronizar", type="primary", use_container_width=True):
        with st.spinner("Reamostando e detectando pico…"):
            raw_synced = {}
            fs_info    = {}
            msgs_pre   = []
            for fname, df in files_data.items():
                r, fs_orig, desc = resample_to_regular(df, fs_target)
                raw_synced[fname]  = r
                fs_info[fname]     = fs_orig
                msgs_pre.append(f"**{fname[:35]}**: {desc}")

            st.session_state.raw_synced = raw_synced
            st.session_state.target_fs  = fs_target
            st.session_state.fs_info    = fs_info
            st.session_state.proc_data          = {}
            st.session_state.proc_data_nofilter = {}

            janela_samp = int(janela_seg * fs_target)
            offsets     = {kinem_ref: 0}
            msgs_sync   = []

            s_k_l5 = try_numeric(raw_synced[kinem_ref][l5_kinem_col])
            peak_k = find_highest_peak(s_k_l5, janela_samp, fs_target)
            st.session_state.peak_ref = peak_k
            st.session_state.synced   = True
            st.session_state.show_preview = False
            msgs_sync.append(f"**Kinem L5** — pico @ {peak_k} ({peak_k/fs_target:.2f} s) → x=0")

            win = int(1.0 * fs_target)
            s_k_knee  = try_numeric(raw_synced[kinem_ref][knee_kinem_col])
            k_start   = max(0, peak_k - win)
            k_end     = min(len(s_k_knee), peak_k + win)
            peak_knee = find_highest_peak(s_k_knee.iloc[k_start:k_end].reset_index(drop=True), k_end - k_start, fs_target) + k_start
            msgs_sync.append(f"**Kinem Joelho (Côndilo)** — pico @ {peak_knee} ({peak_knee/fs_target:.2f} s) → Δ {(peak_knee-peak_k)/fs_target:+.3f} s")

            if l5_acc != NONE and l5_acc_col and l5_acc_col in raw_synced.get(l5_acc, pd.DataFrame()).columns:
                p = find_sync_xcorr(
                    raw_synced[kinem_ref][l5_kinem_col],
                    raw_synced[l5_acc][l5_acc_col],
                    peak_k, janela_samp, fs_target,
                )
                offsets[l5_acc] = peak_k - p
                msgs_sync.append(f"**L5 ACC** — pico @ {p} ({p/fs_target:.2f} s) → offset {peak_k-p:+d}")
                if l5_gyr != NONE:
                    offsets[l5_gyr] = peak_k - p
                    msgs_sync.append(f"**L5 GYR** — offset {peak_k-p:+d} (= ACC L5)")

            if knee_acc != NONE and knee_acc_col and knee_acc_col in raw_synced.get(knee_acc, pd.DataFrame()).columns:
                p = find_sync_xcorr(
                    raw_synced[kinem_ref][knee_kinem_col],
                    raw_synced[knee_acc][knee_acc_col],
                    peak_knee, janela_samp, fs_target,
                )
                offsets[knee_acc] = peak_knee - p
                msgs_sync.append(f"**Joelho ACC** — pico @ {p} ({p/fs_target:.2f} s) → offset {peak_knee-p:+d}")
                if knee_gyr != NONE:
                    offsets[knee_gyr] = peak_knee - p
                    msgs_sync.append(f"**Joelho GYR** — offset {peak_knee-p:+d} (= ACC Joelho)")

            for fname in file_names:
                if fname not in offsets:
                    offsets[fname] = 0
            st.session_state.offsets = offsets
            st.session_state.synced_l5_col   = l5_kinem_col
            st.session_state.synced_knee_col = knee_kinem_col

            with st.expander("📋 Detalhes da sincronização", expanded=False):
                st.markdown("**Frequências detectadas:**")
                for m in msgs_pre: st.write(m)
                st.markdown("**Offsets calculados:**")
                for m in msgs_sync: st.write(m)

# ──────────────────────────────────────────────
# Auto-resync quando coluna de referência muda
# ──────────────────────────────────────────────
if (st.session_state.synced
        and st.session_state.raw_synced
        and st.session_state.peak_ref is not None):
    _col_l5_changed   = st.session_state.synced_l5_col   != l5_kinem_col
    _col_knee_changed = st.session_state.synced_knee_col != knee_kinem_col
    if _col_l5_changed or _col_knee_changed:
        _raws  = st.session_state.raw_synced
        _tfs   = st.session_state.target_fs or 100
        _jsamp = int(janela_seg * _tfs)
        _offs  = dict(st.session_state.offsets)   # cópia dos offsets atuais

        if _col_l5_changed and l5_kinem_col in _raws.get(kinem_ref, pd.DataFrame()).columns:
            _s_l5 = try_numeric(_raws[kinem_ref][l5_kinem_col])
            _pk   = find_highest_peak(_s_l5, _jsamp, _tfs)
            st.session_state.peak_ref       = _pk
            st.session_state.synced_l5_col  = l5_kinem_col
            _offs[kinem_ref] = 0
            # re-sync ACC L5 com novo peak_k
            if (l5_acc != NONE and l5_acc_col
                    and l5_acc_col in _raws.get(l5_acc, pd.DataFrame()).columns):
                _p = find_sync_xcorr(_raws[kinem_ref][l5_kinem_col],
                                     _raws[l5_acc][l5_acc_col],
                                     _pk, _jsamp, _tfs)
                _offs[l5_acc] = _pk - _p
                if l5_gyr != NONE:
                    _offs[l5_gyr] = _pk - _p

        _pk_l5 = st.session_state.peak_ref
        if knee_kinem_col in _raws.get(kinem_ref, pd.DataFrame()).columns:
            _win   = int(1.0 * _tfs)
            _sk    = try_numeric(_raws[kinem_ref][knee_kinem_col])
            _ks    = max(0, _pk_l5 - _win)
            _ke    = min(len(_sk), _pk_l5 + _win)
            _pk_kn = find_highest_peak(
                _sk.iloc[_ks:_ke].reset_index(drop=True), _ke - _ks, _tfs
            ) + _ks
            if (knee_acc != NONE and knee_acc_col
                    and knee_acc_col in _raws.get(knee_acc, pd.DataFrame()).columns):
                _p = find_sync_xcorr(_raws[kinem_ref][knee_kinem_col],
                                     _raws[knee_acc][knee_acc_col],
                                     _pk_kn, _jsamp, _tfs)
                _offs[knee_acc] = _pk_kn - _p
                if knee_gyr != NONE:
                    _offs[knee_gyr] = _pk_kn - _p
            st.session_state.synced_knee_col = knee_kinem_col

        st.session_state.offsets   = _offs
        st.session_state.proc_data = {}          # força reprocessamento

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
# Verificação de alinhamento
# ──────────────────────────────────────────────
if st.session_state.synced and st.session_state.raw_synced and st.session_state.peak_ref is not None:
    _vfs = st.session_state.target_fs or 100
    _vraw, _vx_samp, _ = get_aligned_data(
        st.session_state.raw_synced,
        st.session_state.offsets,
        st.session_state.peak_ref,
        ref_file=kinem_ref,
    )
    if _vraw is None:
        _vraw    = {f: df.copy() for f, df in st.session_state.raw_synced.items()}
        _vx_samp = np.arange(max(len(d) for d in _vraw.values())) - st.session_state.peak_ref
    _vx = _vx_samp / _vfs

    def _render_verif(title, kinem_col, phone_file, phone_col, label_k, label_p):
        check = []
        df_k = _vraw.get(kinem_ref, pd.DataFrame())
        if kinem_col in df_k.columns:
            check.append((df_k, kinem_col, label_k))
        if phone_file != NONE and phone_col and phone_file in _vraw:
            df_p = _vraw[phone_file]
            if phone_col in df_p.columns:
                check.append((df_p, phone_col, label_p))
        if len(check) < 2:
            return
        with st.expander(f"🔍 Verificação — alinhamento {title}", expanded=True):
            colors_v = ["blue", "red"]
            series = []
            caps = []
            for df_s, col, lbl in check:
                s = try_numeric(df_s[col]).fillna(0).values.astype(float)
                pk = np.nanmax(np.abs(s))
                series.append((s / pk if pk > 0 else s, lbl))
                caps.append(f"`{col}`")
            cap = "  |  ".join(
                f"{'🔵' if i==0 else '🔴'} **{series[i][1]}**: {caps[i]}"
                for i in range(len(series))
            )
            st.caption(cap + f"  ·  reamostrado a {_vfs:.0f} Hz  ·  normalizado pelo pico  ·  sem filtro passa-baixa")
            fig_v = go.Figure()
            for i, (s_n, lbl) in enumerate(series):
                fig_v.add_trace(go.Scatter(x=_vx, y=s_n, mode="lines",
                    line=dict(color=colors_v[i], width=2), name=lbl, opacity=0.85))
            if len(series) == 2:
                fig_v.add_trace(go.Scatter(x=_vx, y=series[0][0]-series[1][0], mode="lines",
                    line=dict(color="gray", width=1, dash="dot"), name="Diferença"))
            fig_v.add_vline(x=0, line_dash="dash", line_color="black",
                            annotation_text="salto", annotation_position="top right")
            fig_v.update_layout(
                title=f"{title} — normalizado pelo pico (sem filtro)",
                xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[-5, 5]),
                yaxis=dict(title="Amplitude norm."),
                hovermode="x unified", template="plotly_white", height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_v, use_container_width=True,
                            key=f"verif_{title}_{kinem_col}_{phone_col}")

    _render_verif("L5",     l5_kinem_col,   l5_acc,   l5_acc_col,   "Kinem L5",     "ACC L5")
    _render_verif("Joelho", knee_kinem_col, knee_acc, knee_acc_col, "Kinem Joelho", "ACC Joelho")

    # ──────────────────────────────────────────────
    # Processamento inline
    # ──────────────────────────────────────────────
    st.divider()
    proc_done = bool(st.session_state.proc_data)
    with st.expander(
        "⚙️ Processamento  ✔ Aplicado" if proc_done else "⚙️ Processamento  ← Configure e processe aqui",
        expanded=not proc_done,
    ):
        pc1, pc2 = st.columns(2)
        with pc1:
            do_detrend = st.checkbox("Detrend (remover tendência linear)", value=True, key="do_detrend")
        with pc2:
            do_lowpass = st.checkbox("Filtro passa-baixa (Butterworth)", value=True, key="do_lowpass")

        if do_lowpass:
            fl1, fl2 = st.columns(2)
            with fl1:
                cutoff_hz = st.number_input(
                    "Frequência de corte (Hz)",
                    min_value=0.1, max_value=float(fs_target // 2),
                    value=min(20.0, float(fs_target // 2 - 1)), step=0.5,
                    key="cutoff_hz",
                )
            with fl2:
                filt_order = st.selectbox("Ordem do filtro", [2, 4, 6, 8], index=1, key="filt_order")
        else:
            cutoff_hz  = 20.0
            filt_order = 4

        if st.button("🔧 Processar", type="primary", use_container_width=True, key="btn_processar"):
            raw = st.session_state.raw_synced
            proc, proc_nofilter = {}, {}
            for fname, df in raw.items():
                r = df.copy()
                if do_detrend:
                    r = apply_detrend(r)
                proc_nofilter[fname] = r.copy()
                if do_lowpass:
                    r = apply_lowpass(r, fs_target, cutoff_hz, filt_order)
                proc[fname] = r
            st.session_state.proc_data          = proc
            st.session_state.proc_data_nofilter = proc_nofilter
            st.rerun()

# ──────────────────────────────────────────────
# Auto-visualização — todos os eixos X, Y, Z
# ──────────────────────────────────────────────
if st.session_state.proc_data and st.session_state.synced:
    _pfs = st.session_state.target_fs or 100

    aligned_data, x_samp, align_msg = get_aligned_data(
        st.session_state.proc_data,
        st.session_state.offsets,
        st.session_state.peak_ref,
        ref_file=kinem_ref,
    )
    if aligned_data is None:
        st.error(align_msg)
        st.stop()

    x_axis      = x_samp / _pfs
    x_min_data  = float(x_axis.min())
    x_max_data  = float(x_axis.max())

    # ── Coleta automática de colunas ──
    kdf = aligned_data.get(kinem_ref, pd.DataFrame())
    l5_kinem_cols   = kinem_cols_for_body(kdf, "l5", "l 5")
    knee_kinem_cols = kinem_cols_for_body(kdf, "condilo", "joelho", "knee")

    def get_phone_xyz(fname):
        if fname == NONE or fname not in aligned_data:
            return []
        return [c for c in aligned_data[fname].columns if is_xyz_col(c)]

    l5_acc_xyz   = get_phone_xyz(l5_acc)
    l5_gyr_xyz   = get_phone_xyz(l5_gyr)
    knee_acc_xyz = get_phone_xyz(knee_acc)
    knee_gyr_xyz = get_phone_xyz(knee_gyr)

    def make_auto_traces(kinem_cols, acc_fname, acc_xyz, gyr_fname, gyr_xyz):
        traces = []
        for col in kinem_cols:
            if col in kdf.columns:
                traces.append((kinem_ref, col, try_numeric(kdf[col])))
        if acc_fname != NONE and acc_fname in aligned_data:
            for col in acc_xyz:
                traces.append((acc_fname, col, try_numeric(aligned_data[acc_fname][col])))
        if gyr_fname != NONE and gyr_fname in aligned_data:
            for col in gyr_xyz:
                traces.append((gyr_fname, col, try_numeric(aligned_data[gyr_fname][col])))
        return traces

    l5_traces   = make_auto_traces(l5_kinem_cols,   l5_acc,   l5_acc_xyz,   l5_gyr,   l5_gyr_xyz)
    knee_traces = make_auto_traces(knee_kinem_cols, knee_acc, knee_acc_xyz, knee_gyr, knee_gyr_xyz)

    st.divider()
    st.subheader("📊 Sinais sincronizados — todos os eixos X, Y, Z")
    st.caption(align_msg)

    # Mostra o sinal completo sincronizado
    viz_xmin = x_min_data
    viz_xmax = x_max_data

    def render_auto_charts(traces):
        for fname, col, y in traces:
            dcol = display_col_name(fname, col, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr)
            fig_i = go.Figure()
            fig_i.add_trace(go.Scatter(
                x=x_axis, y=y, mode="lines",
                line=dict(width=1.5), showlegend=False,
            ))
            fig_i.add_vline(x=0, line_dash="dash", line_color="gray",
                            annotation_text="salto", annotation_position="top right")
            fig_i.update_layout(
                title=dict(text=f"<b>{fname[:28]}</b> · {dcol}", font_size=12),
                xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[viz_xmin, viz_xmax]),
                yaxis_title="", height=230,
                margin=dict(t=42, b=38, l=55, r=10),
                hovermode="x", template="plotly_white",
            )
            st.plotly_chart(fig_i, use_container_width=True)

    auto_c1, auto_c2 = st.columns(2)
    with auto_c1:
        st.markdown("#### 🟢 L5")
        render_auto_charts(l5_traces)
    with auto_c2:
        st.markdown("#### 🟠 Joelho")
        render_auto_charts(knee_traces)

    st.divider()

    # ──────────────────────────────────────────────
    # Seleção de janela
    # ──────────────────────────────────────────────
    st.subheader("🪟 Seleção de janela")
    wc1, wc2 = st.columns(2)
    with wc1:
        view_start = st.number_input(
            "Início (s) relativo ao pico",
            value=float(max(x_min_data, -2.0)), step=0.5, key="view_start",
        )
    with wc2:
        view_end = st.number_input(
            "Fim (s) relativo ao pico",
            value=float(min(x_max_data, 8.0)), step=0.5, key="view_end",
        )

    st.divider()

    # ──────────────────────────────────────────────
    # Check de qualidade
    # ──────────────────────────────────────────────
    with st.expander("⚙️ Colunas para check de qualidade (1 por fonte)", expanded=False):
        st.caption("Escolha exatamente qual coluna usar de cada fonte. Os sinais serão plotados sobrepostos (z-score).")
        qk1, qk2 = st.columns(2)

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
                ]),
            )

        l5_acc_num   = numeric_cols(aligned_data.get(l5_acc,  pd.DataFrame())) if l5_acc  != NONE else []
        l5_gyr_num   = numeric_cols(aligned_data.get(l5_gyr,  pd.DataFrame())) if l5_gyr  != NONE else []
        knee_acc_num = numeric_cols(aligned_data.get(knee_acc, pd.DataFrame())) if knee_acc != NONE else []
        knee_gyr_num = numeric_cols(aligned_data.get(knee_gyr, pd.DataFrame())) if knee_gyr != NONE else []

        with qk1:
            qa_acc_l5_col = st.selectbox(
                "🟢 ACC — L5", l5_acc_num if l5_acc_num else ["—"], key="qa_accl5",
                index=col_default(l5_acc_num, ["z", "y", "x"]) if l5_acc_num else 0,
            ) if l5_acc_num else None
            qa_gyr_l5_col = st.selectbox(
                "🟢 GYR — L5", l5_gyr_num if l5_gyr_num else ["—"], key="qa_gyrl5",
                index=col_default(l5_gyr_num, ["z", "y", "x"]) if l5_gyr_num else 0,
            ) if l5_gyr_num else None
        with qk2:
            qa_acc_knee_col = st.selectbox(
                "🟠 ACC — Joelho", knee_acc_num if knee_acc_num else ["—"], key="qa_accknee",
                index=col_default(knee_acc_num, ["z", "y", "x"]) if knee_acc_num else 0,
            ) if knee_acc_num else None
            qa_gyr_knee_col = st.selectbox(
                "🟠 GYR — Joelho", knee_gyr_num if knee_gyr_num else ["—"], key="qa_gyrknee",
                index=col_default(knee_gyr_num, ["z", "y", "x"]) if knee_gyr_num else 0,
            ) if knee_gyr_num else None

    show_qa = st.checkbox("🔍 Checar qualidade dos dados", value=False)
    if show_qa:
        qa_xmin = view_start
        qa_xmax = view_end
        mask_qa = (x_axis >= qa_xmin) & (x_axis <= qa_xmax)
        x_view  = x_axis[mask_qa]

        def get_qa_entry(fname, col_name):
            df_q = aligned_data.get(fname) if (fname and fname != NONE) else None
            if df_q is None or col_name is None or col_name not in df_q.columns:
                return None
            y = try_numeric(df_q[col_name]).values[mask_qa].astype(float)
            if np.all(np.isnan(y)):
                return None
            dcol = display_col_name(fname, col_name, kinem_ref, l5_acc, l5_gyr, knee_acc, knee_gyr)
            return (float(np.nanstd(y)), f"{fname[:22]} · {dcol}", y)

        e_kl5     = get_qa_entry(kinem_ref, qa_kinem_l5_col)
        e_accl5   = get_qa_entry(l5_acc   if l5_acc   != NONE else "", qa_acc_l5_col)
        e_gyrl5   = get_qa_entry(l5_gyr   if l5_gyr   != NONE else "", qa_gyr_l5_col)
        e_kknee   = get_qa_entry(kinem_ref, qa_kinem_knee_col)
        e_accknee = get_qa_entry(knee_acc if knee_acc != NONE else "", qa_acc_knee_col)
        e_gyrknee = get_qa_entry(knee_gyr if knee_gyr != NONE else "", qa_gyr_knee_col)

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
                        mn, sd = np.nanmean(y_raw), np.nanstd(y_raw)
                        y_norm = (y_raw - mn) / sd if sd > 0 else y_raw - mn
                        fig_qa.add_trace(go.Scatter(
                            x=x_view, y=y_norm, mode="lines",
                            name=f"{lbl}  (σ_orig={std_val:.3f})",
                        ))
                    fig_qa.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="salto")
                    fig_qa.update_layout(
                        xaxis=dict(title="Tempo (s)  —  0 = pico do salto", range=[qa_xmin, qa_xmax]),
                        yaxis_title="z-score",
                        height=380, template="plotly_white", hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                        margin=dict(t=30, b=40),
                    )
                    st.plotly_chart(fig_qa, use_container_width=True)

    st.divider()

    # ──────────────────────────────────────────────
    # Exportar Excel — apenas janela selecionada
    # ──────────────────────────────────────────────
    st.subheader("📥 Exportar Excel")
    st.caption(
        f"Exporta todos os eixos X, Y, Z • janela: **{view_start:+.1f} s → {view_end:+.1f} s** relativo ao pico"
    )

    if st.button("Gerar arquivo Excel (L5 + Joelho)", use_container_width=True):
        mask_exp = (x_axis >= view_start) & (x_axis <= view_end)
        win_idx  = np.where(mask_exp)[0]

        if len(win_idx) == 0:
            st.error("Janela vazia — ajuste os limites de início/fim.")
        else:
            windowed = {
                fname: df.iloc[win_idx].reset_index(drop=True)
                for fname, df in aligned_data.items()
            }
            t_w = np.arange(len(win_idx)) / _pfs   # tempo começa em 0

            df_l5   = build_export_sheet(windowed, kinem_ref, l5_acc,   l5_gyr,
                                          ["l5", "l 5"],                 t_w, _pfs)
            df_knee = build_export_sheet(windowed, kinem_ref, knee_acc, knee_gyr,
                                          ["condilo", "joelho", "knee"], t_w, _pfs)

            _buf = io.BytesIO()
            with pd.ExcelWriter(_buf, engine="openpyxl") as _writer:
                df_l5.to_excel(_writer,   sheet_name="L5",     index=False)
                df_knee.to_excel(_writer, sheet_name="Joelho", index=False)
            _buf.seek(0)
            st.download_button(
                "⬇ Baixar sinais_sincronizados.xlsx",
                _buf,
                file_name="sinais_sincronizados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
