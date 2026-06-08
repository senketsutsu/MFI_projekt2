"""Interaktywny dashboard regresji wielomianowej."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from regression import fit_polynomial_regression

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_LEARNING_RATE = 0.0001


@st.cache_data(show_spinner=False)
def list_datasets():
    return sorted(p.stem for p in DATA_DIR.glob("*.csv"))


@st.cache_data(show_spinner="Wczytywanie danych…")
def load_dataset(name: str):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df.iloc[:, 0].values, df.iloc[:, 1].values


@st.cache_data(show_spinner="Trenowanie modelu…")
def train_model(dataset: str, degree: int, learning_rate: float):
    x, y = load_dataset(dataset)
    result = fit_polynomial_regression(x, y, degree, learning_rate=learning_rate)
    return x, y, result


@st.cache_data(show_spinner="Liczenie eksperymentów (jednorazowo, ~2–3 min)…")
def compute_experiments(learning_rate: float):
    rows = []
    for name in list_datasets():
        x, y = load_dataset(name)
        for degree in range(1, 11):
            result = fit_polynomial_regression(x, y, degree, learning_rate=learning_rate)
            rows.append(
                {
                    "dataset": name,
                    "degree": degree,
                    "loss": result["final_loss"],
                    "iterations": result["iterations"],
                }
            )
    return pd.DataFrame(rows)


def plot_fit(x, y, y_pred, dataset: str, degree: int):
    order = np.argsort(x)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x[order],
            y=y[order],
            mode="markers",
            name="Dane rzeczywiste",
            marker=dict(size=4, opacity=0.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[order],
            y=y_pred[order],
            mode="lines",
            name=f"Model (stopień {degree})",
            line=dict(width=2),
        )
    )
    fig.update_layout(
        title=f"Dopasowanie modelu – {dataset}",
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=450,
    )
    return fig


def plot_loss_history(loss_history):
    fig = px.line(
        y=loss_history,
        labels={"index": "Iteracja", "value": "Wartość funkcji straty"},
        title="Historia funkcji straty podczas uczenia",
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_experiments_for_dataset(experiment_df: pd.DataFrame, dataset: str, current_degree: int):
    subset = experiment_df[experiment_df["dataset"] == dataset]
    fig = px.line(
        subset,
        x="degree",
        y="loss",
        markers=True,
        log_y=True,
        title=f"MSE vs stopień wielomianu – {dataset}",
        labels={"degree": "Stopień wielomianu", "loss": "MSE (skala log)"},
    )
    best = subset.loc[subset["loss"].idxmin()]
    fig.add_vline(x=current_degree, line_dash="dash", line_color="orange", annotation_text="wybrany")
    fig.add_vline(
        x=best["degree"],
        line_dash="dot",
        line_color="green",
        annotation_text=f"najlepszy ({int(best['degree'])})",
    )
    fig.update_layout(height=400)
    return fig


def main():
    st.set_page_config(
        page_title="Regresja wielomianowa",
        page_icon="📈",
        layout="wide",
    )

    st.title("Dashboard – regresja wielomianowa (gradient descent)")
    st.caption(
        "Interaktywna wizualizacja modelu z własnej implementacji algorytmu najszybszego spadku."
    )

    datasets = list_datasets()
    if not datasets:
        st.error("Brak plików CSV w katalogu data/.")
        st.stop()

    with st.sidebar:
        st.header("Parametry modelu")
        dataset = st.selectbox("Zbiór danych", datasets)
        degree = st.slider("Stopień wielomianu", min_value=1, max_value=10, value=5)
        learning_rate = st.number_input(
            "Współczynnik uczenia",
            min_value=1e-6,
            max_value=1.0,
            value=DEFAULT_LEARNING_RATE,
            format="%.6f",
            step=1e-5,
        )
        st.divider()
        st.markdown(
            "**Wskazówka:** wyższy stopień = bardziej złożony model. "
            "Dla trudnych zbiorów (np. `noisy_polynomial`) używamy małego learning rate."
        )

    x, y, result = train_model(dataset, degree, learning_rate)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE (strata końcowa)", f"{result['final_loss']:.4g}")
    col2.metric("Iteracje", result["iterations"])
    col3.metric("Liczba punktów", len(x))
    col4.metric("Parametry θ", len(result["theta"]))

    tab_fit, tab_loss, tab_exp = st.tabs(["Dopasowanie", "Historia straty", "Eksperymenty"])

    with tab_fit:
        st.plotly_chart(plot_fit(x, y, result["y_pred"], dataset, degree), use_container_width=True)

    with tab_loss:
        st.plotly_chart(plot_loss_history(result["loss_history"]), use_container_width=True)

    with tab_exp:
        st.markdown(
            "Porównanie jakości dopasowania dla stopni 1–10 na wszystkich zbiorach. "
            "Pierwsze uruchomienie tej zakładki trwa kilka minut (wynik jest cache'owany)."
        )
        if st.button("Uruchom / odśwież eksperymenty"):
            st.session_state["run_experiments"] = True

        if st.session_state.get("run_experiments"):
            experiment_df = compute_experiments(learning_rate)
            st.plotly_chart(
                plot_experiments_for_dataset(experiment_df, dataset, degree),
                use_container_width=True,
            )

            best_all = (
                experiment_df.loc[experiment_df.groupby("dataset")["loss"].idxmin()]
                .sort_values("dataset")[["dataset", "degree", "loss", "iterations"]]
                .reset_index(drop=True)
            )
            st.subheader("Optymalny stopień per zbiór")
            st.dataframe(best_all, use_container_width=True, hide_index=True)
        else:
            st.info("Kliknij „Uruchom / odśwież eksperymenty”, aby policzyć MSE dla wszystkich stopni.")


if __name__ == "__main__":
    main()
