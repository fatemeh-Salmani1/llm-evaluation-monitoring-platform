from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.monitoring.dashboard_data import (
    build_case_frame,
    build_history_frame,
    load_evaluation_records,
    load_run_history,
    records_path_for_run,
)

DEFAULT_RUN_DIRECTORY = Path(
    "data/processed/evaluation_runs"
)


def format_score(value: float | None) -> str:
    """Format a normalized score for display."""

    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.3f}"


def calculate_delta(
    current: float | None,
    previous: float | None,
) -> str | None:
    """Format the change from the previous benchmark run."""

    if (
        current is None
        or previous is None
        or pd.isna(current)
        or pd.isna(previous)
    ):
        return None

    return f"{current - previous:+.3f}"


def render_overview(
    history: pd.DataFrame,
) -> None:
    """Render the latest benchmark metrics."""

    latest = history.iloc[-1]
    previous = (
        history.iloc[-2]
        if len(history) > 1
        else None
    )

    columns = st.columns(5)

    columns[0].metric(
        "Success rate",
        f"{latest['success_rate']:.1%}",
        (
            calculate_delta(
                latest["success_rate"],
                previous["success_rate"],
            )
            if previous is not None
            else None
        ),
    )
    columns[1].metric(
        "Retrieval recall",
        format_score(latest["retrieval_recall"]),
        (
            calculate_delta(
                latest["retrieval_recall"],
                previous["retrieval_recall"],
            )
            if previous is not None
            else None
        ),
    )
    columns[2].metric(
        "Fact coverage",
        format_score(latest["fact_coverage"]),
        (
            calculate_delta(
                latest["fact_coverage"],
                previous["fact_coverage"],
            )
            if previous is not None
            else None
        ),
    )
    columns[3].metric(
        "Deterministic score",
        format_score(latest["overall_score"]),
        (
            calculate_delta(
                latest["overall_score"],
                previous["overall_score"],
            )
            if previous is not None
            else None
        ),
    )
    columns[4].metric(
        "LLM judge score",
        format_score(latest["judge_score"]),
        (
            calculate_delta(
                latest["judge_score"],
                previous["judge_score"],
            )
            if previous is not None
            else None
        ),
    )


def render_quality_trends(
    history: pd.DataFrame,
) -> None:
    """Render benchmark quality trends."""

    metric_columns = [
        "retrieval_recall",
        "fact_coverage",
        "citation_validity",
        "overall_score",
    ]

    if history["judge_score"].notna().any():
        metric_columns.append("judge_score")

    trend_data = history[
        [
            "completed_at",
            "run_id",
            *metric_columns,
        ]
    ].melt(
        id_vars=["completed_at", "run_id"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="score",
    )

    labels = {
        "retrieval_recall": "Retrieval recall",
        "fact_coverage": "Fact coverage",
        "citation_validity": "Citation validity",
        "overall_score": "Deterministic score",
        "judge_score": "LLM judge score",
    }

    trend_data["metric"] = trend_data[
        "metric"
    ].map(labels)

    figure = px.line(
        trend_data,
        x="completed_at",
        y="score",
        color="metric",
        markers=True,
        hover_data=["run_id"],
        labels={
            "completed_at": "Completed at",
            "score": "Score",
            "metric": "Metric",
        },
    )
    figure.update_yaxes(
        range=[0, 1.05],
        tickformat=".0%",
    )
    figure.update_layout(
        legend_title_text="",
        hovermode="x unified",
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_latency_trend(
    history: pd.DataFrame,
) -> None:
    """Render average benchmark latency over time."""

    figure = px.bar(
        history,
        x="completed_at",
        y="average_duration_ms",
        hover_data=["run_id", "total_cases"],
        labels={
            "completed_at": "Completed at",
            "average_duration_ms": (
                "Average duration (ms)"
            ),
        },
        color="average_duration_ms",
        color_continuous_scale="Blues",
    )
    figure.update_layout(
        coloraxis_showscale=False,
    )

    st.plotly_chart(
        figure,
        width="stretch",
    )


def render_run_details(
    history: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Render per-case results for a selected run."""

    st.subheader("Run details")

    run_ids = list(
        reversed(history["run_id"].tolist())
    )

    selected_run_id = st.selectbox(
        "Select a benchmark run",
        options=run_ids,
    )

    if selected_run_id is None:
        return

    selected_summary = history.loc[
        history["run_id"] == selected_run_id
    ].iloc[0]

    details = st.columns(4)
    details[0].metric(
        "Cases",
        int(selected_summary["total_cases"]),
    )
    details[1].metric(
        "Successful",
        int(selected_summary["successful_cases"]),
    )
    details[2].metric(
        "Failed",
        int(selected_summary["failed_cases"]),
    )
    details[3].metric(
        "Top K",
        int(selected_summary["top_k"]),
    )

    records_path = records_path_for_run(
        output_directory=output_directory,
        run_id=selected_run_id,
    )

    try:
        records = load_evaluation_records(
            records_path
        )
    except (FileNotFoundError, ValueError) as error:
        st.warning(str(error))
        return

    case_frame = build_case_frame(records)

    score_columns = [
        "case_id",
        "status",
        "category",
        "difficulty",
        "retrieval_recall",
        "fact_coverage",
        "citation_validity",
        "deterministic_score",
        "judge_score",
        "duration_ms",
    ]

    st.dataframe(
        case_frame[score_columns],
        width="stretch",
        hide_index=True,
        column_config={
            "case_id": "Case",
            "status": "Status",
            "category": "Category",
            "difficulty": "Difficulty",
            "retrieval_recall": st.column_config.NumberColumn(
                "Retrieval",
                format="%.3f",
            ),
            "fact_coverage": st.column_config.NumberColumn(
                "Facts",
                format="%.3f",
            ),
            "citation_validity": (
                st.column_config.NumberColumn(
                    "Citations",
                    format="%.3f",
                )
            ),
            "deterministic_score": (
                st.column_config.NumberColumn(
                    "Deterministic",
                    format="%.3f",
                )
            ),
            "judge_score": st.column_config.NumberColumn(
                "Judge",
                format="%.3f",
            ),
            "duration_ms": st.column_config.NumberColumn(
                "Duration (ms)",
                format="%.0f",
            ),
        },
    )

    judged_cases = case_frame[
        case_frame["judge_score"].notna()
    ]

    if judged_cases.empty:
        st.info(
            "This run does not contain LLM judge results."
        )
        return

    st.subheader("LLM judge analysis")

    selected_case_id = st.selectbox(
        "Select a judged case",
        options=judged_cases["case_id"].tolist(),
    )

    selected_case = judged_cases.loc[
        judged_cases["case_id"] == selected_case_id
    ].iloc[0]

    judge_columns = st.columns(5)
    judge_columns[0].metric(
        "Overall",
        format_score(selected_case["judge_score"]),
    )
    judge_columns[1].metric(
        "Relevance",
        int(selected_case["judge_relevance"]),
    )
    judge_columns[2].metric(
        "Completeness",
        int(selected_case["judge_completeness"]),
    )
    judge_columns[3].metric(
        "Groundedness",
        int(selected_case["judge_groundedness"]),
    )
    judge_columns[4].metric(
        "Clarity",
        int(selected_case["judge_clarity"]),
    )

    st.markdown("**Question**")
    st.write(selected_case["question"])

    st.markdown("**Judge reasoning**")
    st.write(selected_case["judge_reasoning"])


def main() -> None:
    """Render the LLM evaluation monitoring dashboard."""

    st.set_page_config(
        page_title="LLM Evaluation Monitor",
        page_icon="📊",
        layout="wide",
    )

    st.title("LLM Evaluation & Monitoring")
    st.caption(
        "Track retrieval quality, grounded-answer performance, "
        "LLM judge scores, regressions, and latency."
    )

    output_directory = Path(
        st.sidebar.text_input(
            "Evaluation run directory",
            value=str(DEFAULT_RUN_DIRECTORY),
        )
    )

    try:
        summaries = load_run_history(
            output_directory
        )
    except (
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
    ) as error:
        st.error(str(error))
        return

    if not summaries:
        st.info(
            "No benchmark summaries were found. "
            "Run the evaluation benchmark first."
        )
        return

    history = build_history_frame(summaries)

    full_run_size = history["total_cases"].max()
    full_history = history.loc[
        history["total_cases"] == full_run_size
    ].reset_index(drop=True)

    st.subheader("Latest full benchmark")
    render_overview(full_history)

    quality_tab, latency_tab, history_tab = st.tabs(
        [
            "Quality trends",
            "Latency",
            "Run history",
        ]
    )

    with quality_tab:
        render_quality_trends(full_history)

    with latency_tab:
        render_latency_trend(full_history)

    with history_tab:
        st.dataframe(
            history.sort_values(
                "completed_at",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )

    render_run_details(
        history=history,
        output_directory=output_directory,
    )


if __name__ == "__main__":
    main()