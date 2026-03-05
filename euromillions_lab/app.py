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

st.set_page_config(page_title="Euromillones Lab", layout="wide")
st.title("Euromillones Lab (educativo y personal, solo histórico)")
st.caption("Análisis histórico y backtesting walk-forward sin recomendaciones de apuesta.")

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "normalized_df" not in st.session_state:
    st.session_state.normalized_df = None

st.header("A) Cargar datos")
uploaded = st.file_uploader("Sube el XLS/XLSX histórico", type=["xls", "xlsx"])
local_path = st.text_input("O ruta local del archivo", value="")

load_clicked = st.button("Cargar archivo")
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
            st.success(f"Archivo cargado. Hojas detectadas: {', '.join(sheets)}")
    except Exception as exc:
        st.error(f"Error al cargar XLS: {exc}")

raw_df = st.session_state.raw_df
if raw_df is not None:
    st.subheader("Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)

    auto = auto_detect_mapping(raw_df)
    st.subheader("Mapeo de columnas")

    cols = list(raw_df.columns)

    if auto is not None:
        st.info("Mapeo automático detectado. Puedes ajustarlo manualmente si deseas.")
        default_date = auto.date_col
        default_nums = auto.number_cols
        default_stars = auto.star_cols
    else:
        st.warning("No se pudo detectar mapeo automático completo. Indica mapeo manual.")
        default_date = cols[0]
        default_nums = cols[:5] if len(cols) >= 5 else cols
        default_stars = cols[5:7] if len(cols) >= 7 else cols[:2]

    date_col = st.selectbox("date column", cols, index=cols.index(default_date) if default_date in cols else 0)

    num_choices = st.multiselect(
        "5 number columns",
        options=cols,
        default=default_nums if len(default_nums) <= 5 else default_nums[:5],
        max_selections=5,
    )
    star_choices = st.multiselect(
        "2 star columns",
        options=cols,
        default=default_stars if len(default_stars) <= 2 else default_stars[:2],
        max_selections=2,
    )

    if st.button("Normalizar y guardar CSV"):
        if len(num_choices) != 5 or len(star_choices) != 2:
            st.error("Debes seleccionar exactamente 5 columnas de números y 2 de estrellas.")
        else:
            try:
                mapping = ColumnMapping(date_col=date_col, number_cols=num_choices, star_cols=star_choices)
                normalized = normalize_draws(raw_df, mapping)
                st.session_state.normalized_df = normalized
                path = save_normalized_csv(normalized, NORMALIZED_CSV)
                st.success(f"CSV normalizado guardado en: {path}")
                st.dataframe(normalized.head(20), use_container_width=True)
            except Exception as exc:
                st.error(f"Error durante normalización: {exc}")

if st.session_state.normalized_df is None and NORMALIZED_CSV.exists():
    try:
        st.session_state.normalized_df = pd.read_csv(NORMALIZED_CSV, parse_dates=["date"])
        st.info("Se cargó el CSV normalizado existente desde data/processed.")
    except Exception as exc:
        st.warning(f"No se pudo cargar CSV normalizado existente: {exc}")

norm_df = st.session_state.normalized_df

st.header("B) Configurar experimento")
strategy_name = st.selectbox("Estrategia", ["Random", "Frequency", "DecileBucket", "Birthday"])
seed = st.number_input("seed", min_value=0, value=42, step=1)
window_k = st.number_input("window_k (si aplica)", min_value=1, value=200, step=1)
n_sim = st.number_input("N simulaciones baseline", min_value=100, value=5000, step=100)

birthday_dates = []
manual_numbers = []
use_year_mod = False
if strategy_name == "Birthday":
    st.markdown("**Parámetros BirthdayStrategy**")
    dates_text = st.text_area(
        "Fechas (una por línea o separadas por coma, formato YYYY-MM-DD)",
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

st.header("C) Ejecutar backtest")
run_btn = st.button("Run")

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

            st.subheader("Resultados (últimas 50 filas)")
            st.dataframe(results.tail(50), use_container_width=True)

            agg = aggregate_metrics(results)
            baseline = random_baseline_stats(results, n_simulations=int(n_sim), seed=int(seed))

            c1, c2, c3 = st.columns(3)
            c1.metric("Media matches_numbers", f"{agg['mean_matches_numbers']:.3f}")
            c2.metric("Media matches_stars", f"{agg['mean_matches_stars']:.3f}")
            c3.metric("Percentil global vs baseline", f"{baseline['overall_percentile']:.2f}")

            st.write("Distribución matches_numbers:", agg["distribution_matches_numbers"])
            st.write("Distribución matches_stars:", agg["distribution_matches_stars"])
            st.write("Top outcomes (numbers, stars):", agg["top_outcomes"])
            st.write(
                {
                    "expected_mean_numbers": baseline["expected_mean_numbers"],
                    "expected_mean_stars": baseline["expected_mean_stars"],
                }
            )

            fig1, ax1 = plt.subplots()
            ax1.hist(results["matches_numbers"], bins=range(0, 7), align="left", rwidth=0.8)
            ax1.set_title("Histograma matches_numbers")
            ax1.set_xlabel("Aciertos números")
            ax1.set_ylabel("Frecuencia")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            ax2.hist(results["matches_stars"], bins=range(0, 4), align="left", rwidth=0.8)
            ax2.set_title("Histograma matches_stars")
            ax2.set_xlabel("Aciertos estrellas")
            ax2.set_ylabel("Frecuencia")
            st.pyplot(fig2)

            fig3, ax3 = plt.subplots()
            ax3.plot(pd.to_datetime(results["date"]), results["matches_numbers"], marker="o", linewidth=1)
            ax3.set_title("Serie temporal matches_numbers")
            ax3.set_xlabel("Fecha")
            ax3.set_ylabel("Aciertos números")
            st.pyplot(fig3)

            per_draw = baseline.get("percentile_of_strategy_by_draw", [])
            if per_draw:
                fig4, ax4 = plt.subplots()
                ax4.plot(pd.to_datetime(results["date"]), per_draw, linewidth=1)
                ax4.set_title("Percentil vs baseline por sorteo")
                ax4.set_xlabel("Fecha")
                ax4.set_ylabel("Percentil")
                st.pyplot(fig4)

            st.download_button(
                "Descargar results.csv",
                data=RESULTS_CSV.read_bytes(),
                file_name="results.csv",
                mime="text/csv",
            )
            st.success(f"Backtest completado. Archivo guardado: {RESULTS_CSV}")
        except Exception as exc:
            st.error(f"Error en ejecución del backtest: {exc}")
