from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.backtest import run_backtest, save_results
from src.ingest import ColumnMapping, auto_detect_mapping, load_excel_any, normalize_draws, save_normalized_csv
from src.metrics import aggregate_metrics, random_baseline_stats
from src.strategies import BirthdayStrategy, DecileBucketStrategy, FrequencyStrategy, RandomStrategy
from src.utils import ensure_project_dirs

BASE_DIR = Path(__file__).resolve().parent
ensure_project_dirs(BASE_DIR)

NORMALIZED_CSV = BASE_DIR / "data" / "processed" / "euromillones_normalized.csv"
RESULTS_CSV = BASE_DIR / "outputs" / "results.csv"

st.set_page_config(page_title="Euromillones Lab", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #060916;
        --panel: #11172b;
        --panel-soft: #131d38;
        --text: #e8ecf8;
        --muted: #a9b6d3;
        --primary: #6ca4ff;
        --accent: #44d6c9;
        --border: rgba(255,255,255,.08);
    }

    .stApp {
        background: radial-gradient(1200px 700px at 10% -20%, #1a2a55 0%, var(--bg) 45%);
        color: var(--text);
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1250px;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    div[data-testid="stVerticalBlock"] div.section-card {
        background: linear-gradient(180deg, var(--panel) 0%, var(--panel-soft) 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .hero {
        padding: 1.4rem 1.5rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(108,164,255,0.18), rgba(68,214,201,0.10));
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2rem;
        line-height: 1.2;
    }

    .hero p {
        margin: 0.6rem 0 0;
        color: var(--muted);
        font-size: 1rem;
    }

    .mini-note {
        font-size: 0.9rem;
        color: var(--muted);
    }

    .metric-card {
        padding: .65rem .8rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(255,255,255,.02);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(108,164,255,.35);
        background: linear-gradient(180deg, #2d4f96 0%, #1f3d7b 100%);
        color: white;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: rgba(108,164,255,.85);
        box-shadow: 0 0 0 2px rgba(108,164,255,.2);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.02);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: .6rem .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎯 Euromillones Lab</h1>
      <p>Analiza sorteos históricos y ejecuta backtesting walk-forward con una interfaz más clara, moderna y rápida de usar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Uso educativo y personal. No es una recomendación de apuesta.")

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "normalized_df" not in st.session_state:
    st.session_state.normalized_df = None
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "baseline" not in st.session_state:
    st.session_state.baseline = None
if "agg" not in st.session_state:
    st.session_state.agg = None

with st.sidebar:
    st.markdown("### ⚙️ Panel rápido")
    st.markdown("""
    1. Carga archivo histórico.  
    2. Revisa/ajusta mapeo de columnas.  
    3. Ejecuta el experimento y compara contra baseline aleatorio.
    """)
    st.divider()
    st.markdown(f"**CSV normalizado:** `{NORMALIZED_CSV}`")
    st.markdown(f"**Resultados:** `{RESULTS_CSV}`")

if st.session_state.normalized_df is None and NORMALIZED_CSV.exists():
    try:
        st.session_state.normalized_df = pd.read_csv(NORMALIZED_CSV, parse_dates=["date"])
        st.toast("Se cargó automáticamente el CSV normalizado existente.")
    except Exception as exc:
        st.warning(f"No se pudo cargar CSV normalizado existente: {exc}")

setup_tab, run_tab, insights_tab = st.tabs(["📥 Datos", "🧪 Backtest", "📊 Insights"])

with setup_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("A) Cargar datos")
    c1, c2 = st.columns([1, 1])
    with c1:
        uploaded = st.file_uploader("Sube el XLS/XLSX histórico", type=["xls", "xlsx"])
    with c2:
        local_path = st.text_input("O usa una ruta local", value="")

    load_clicked = st.button("Cargar archivo", use_container_width=True)
    if load_clicked:
        try:
            if uploaded is not None:
                raw_df, sheets = load_excel_any(uploaded)
            elif local_path.strip():
                raw_df, sheets = load_excel_any(local_path.strip())
            else:
                st.warning("Selecciona un archivo o indica una ruta local.")
                raw_df, sheets = None, []

            if raw_df is not None:
                st.session_state.raw_df = raw_df
                st.success(f"Archivo cargado correctamente. Hojas detectadas: {', '.join(sheets)}")
        except Exception as exc:
            st.error(f"Error al cargar XLS: {exc}")

    raw_df = st.session_state.raw_df
    if raw_df is not None:
        st.markdown("<p class='mini-note'>Vista previa de los primeros 20 registros.</p>", unsafe_allow_html=True)
        st.dataframe(raw_df.head(20), use_container_width=True, height=320)

        auto = auto_detect_mapping(raw_df)
        st.subheader("B) Mapeo de columnas")

        cols = list(raw_df.columns)

        if auto is not None:
            st.info("Mapeo automático detectado. Puedes afinarlo manualmente si quieres.")
            default_date = auto.date_col
            default_nums = auto.number_cols
            default_stars = auto.star_cols
        else:
            st.warning("No se detectó mapeo completo. Ajusta manualmente.")
            default_date = cols[0]
            default_nums = cols[:5] if len(cols) >= 5 else cols
            default_stars = cols[5:7] if len(cols) >= 7 else cols[:2]

        m1, m2, m3 = st.columns([1.2, 1.4, 1])
        with m1:
            date_col = st.selectbox("Columna de fecha", cols, index=cols.index(default_date) if default_date in cols else 0)
        with m2:
            num_choices = st.multiselect(
                "5 columnas de números",
                options=cols,
                default=default_nums if len(default_nums) <= 5 else default_nums[:5],
                max_selections=5,
            )
        with m3:
            star_choices = st.multiselect(
                "2 columnas de estrellas",
                options=cols,
                default=default_stars if len(default_stars) <= 2 else default_stars[:2],
                max_selections=2,
            )

        if st.button("Normalizar y guardar CSV", use_container_width=True):
            if len(num_choices) != 5 or len(star_choices) != 2:
                st.error("Debes seleccionar exactamente 5 columnas de números y 2 de estrellas.")
            else:
                try:
                    mapping = ColumnMapping(date_col=date_col, number_cols=num_choices, star_cols=star_choices)
                    normalized = normalize_draws(raw_df, mapping)
                    st.session_state.normalized_df = normalized
                    path = save_normalized_csv(normalized, NORMALIZED_CSV)
                    st.success(f"CSV normalizado guardado en: {path}")
                    st.dataframe(normalized.head(20), use_container_width=True, height=320)
                except Exception as exc:
                    st.error(f"Error durante normalización: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

with run_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("C) Configurar experimento")

    norm_df = st.session_state.normalized_df

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        strategy_name = st.selectbox("Estrategia", ["Random", "Frequency", "DecileBucket", "Birthday"])
    with p2:
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
    with p3:
        window_k = st.number_input("Window k", min_value=1, value=200, step=1)
    with p4:
        n_sim = st.number_input("N simulaciones baseline", min_value=100, value=5000, step=100)

    birthday_dates = []
    manual_numbers = []
    use_year_mod = False
    if strategy_name == "Birthday":
        with st.expander("Parámetros BirthdayStrategy", expanded=True):
            dates_text = st.text_area(
                "Fechas (una por línea o por coma, YYYY-MM-DD)",
                value="",
                help="Ejemplo: 1990-05-12, 1985-10-03",
            )
            manual_text = st.text_input("Números manuales (separados por coma)", value="")
            use_year_mod = st.checkbox("Usar año % 50 como candidato", value=True)

            if dates_text.strip():
                raw_tokens = [t.strip() for chunk in dates_text.splitlines() for t in chunk.split(",") if t.strip()]
                for token in raw_tokens:
                    dt = pd.to_datetime(token, errors="coerce")
                    if pd.notna(dt):
                        birthday_dates.append(dt)

            if manual_text.strip():
                for token in manual_text.split(","):
                    token = token.strip()
                    if token.isdigit():
                        manual_numbers.append(int(token))

    if norm_df is not None and not norm_df.empty:
        min_d, max_d = pd.to_datetime(norm_df["date"]).min(), pd.to_datetime(norm_df["date"]).max()
        start_date, end_date = st.date_input("Rango de fechas", value=(min_d.date(), max_d.date()))
    else:
        start_date = end_date = None
        st.warning("Aún no hay datos normalizados para configurar rango.")

    st.subheader("D) Ejecutar")
    run_btn = st.button("▶ Ejecutar backtest", use_container_width=True)

    if run_btn:
        if norm_df is None or norm_df.empty:
            st.error("Debes cargar y normalizar datos primero.")
        else:
            try:
                if strategy_name == "Random":
                    strategy = RandomStrategy()
                elif strategy_name == "Frequency":
                    strategy = FrequencyStrategy(window_k=int(window_k))
                elif strategy_name == "DecileBucket":
                    strategy = DecileBucketStrategy(window_k=int(window_k))
                else:
                    strategy = BirthdayStrategy(
                        dates_input=birthday_dates,
                        manual_numbers=manual_numbers,
                        use_year_mod=use_year_mod,
                    )

                results = run_backtest(
                    draws_df=norm_df,
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    window_k=int(window_k),
                    seed=int(seed),
                )
                save_results(results, RESULTS_CSV)

                agg = aggregate_metrics(results)
                baseline = random_baseline_stats(results, n_simulations=int(n_sim), seed=int(seed))

                st.session_state.results_df = results
                st.session_state.agg = agg
                st.session_state.baseline = baseline
                st.success(f"Backtest completado. Archivo guardado: {RESULTS_CSV}")
            except Exception as exc:
                st.error(f"Error en ejecución del backtest: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)

with insights_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Resultados e insights")

    results = st.session_state.results_df
    agg = st.session_state.agg
    baseline = st.session_state.baseline

    if results is None or agg is None or baseline is None:
        st.info("Ejecuta un backtest para mostrar métricas y visualizaciones.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Media matches_numbers", f"{agg['mean_matches_numbers']:.3f}")
        m2.metric("Media matches_stars", f"{agg['mean_matches_stars']:.3f}")
        m3.metric("Percentil global vs baseline", f"{baseline['overall_percentile']:.2f}")

        st.markdown("#### Últimas 50 ejecuciones")
        st.dataframe(results.tail(50), use_container_width=True, height=320)

        info1, info2, info3 = st.columns(3)
        with info1:
            st.markdown("<div class='metric-card'>Distribución números</div>", unsafe_allow_html=True)
            st.write(agg["distribution_matches_numbers"])
        with info2:
            st.markdown("<div class='metric-card'>Distribución estrellas</div>", unsafe_allow_html=True)
            st.write(agg["distribution_matches_stars"])
        with info3:
            st.markdown("<div class='metric-card'>Top outcomes</div>", unsafe_allow_html=True)
            st.write(agg["top_outcomes"])

        st.markdown("#### Comparativa con baseline")
        st.write(
            {
                "expected_mean_numbers": baseline["expected_mean_numbers"],
                "expected_mean_stars": baseline["expected_mean_stars"],
            }
        )

        g1, g2 = st.columns(2)

        with g1:
            fig1, ax1 = plt.subplots()
            ax1.hist(results["matches_numbers"], bins=range(0, 7), align="left", rwidth=0.8)
            ax1.set_title("Histograma matches_numbers")
            ax1.set_xlabel("Aciertos números")
            ax1.set_ylabel("Frecuencia")
            st.pyplot(fig1, use_container_width=True)

        with g2:
            fig2, ax2 = plt.subplots()
            ax2.hist(results["matches_stars"], bins=range(0, 4), align="left", rwidth=0.8)
            ax2.set_title("Histograma matches_stars")
            ax2.set_xlabel("Aciertos estrellas")
            ax2.set_ylabel("Frecuencia")
            st.pyplot(fig2, use_container_width=True)

        fig3, ax3 = plt.subplots()
        ax3.plot(pd.to_datetime(results["date"]), results["matches_numbers"], marker="o", linewidth=1)
        ax3.set_title("Serie temporal matches_numbers")
        ax3.set_xlabel("Fecha")
        ax3.set_ylabel("Aciertos números")
        st.pyplot(fig3, use_container_width=True)

        per_draw = baseline.get("percentile_of_strategy_by_draw", [])
        if per_draw:
            fig4, ax4 = plt.subplots()
            ax4.plot(pd.to_datetime(results["date"]), per_draw, linewidth=1)
            ax4.set_title("Percentil vs baseline por sorteo")
            ax4.set_xlabel("Fecha")
            ax4.set_ylabel("Percentil")
            st.pyplot(fig4, use_container_width=True)

        st.download_button(
            "⬇ Descargar results.csv",
            data=RESULTS_CSV.read_bytes(),
            file_name="results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)
