from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class SimulationData:
    """Wrapper around a simulated dataset with column validation."""

    data: pd.DataFrame

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "adv_id",
        "cpg_id",
        "ctrl_spend_h",
        "ctrl_convs_h",
        "ctrl_spend",
        "ctrl_convs",
        "treat_spend",
        "treat_convs",
    )

    def __post_init__(self) -> None:
        missing = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @classmethod
    def load_csv(cls: type[Self], path: str, **kwargs: object) -> Self:
        """Load a CSV file into a SimulationData instance.

        Parameters
        ----------
        path:
            Path to the CSV file.
        **kwargs:
            Extra keyword arguments forwarded to ``pd.read_csv``.
        """
        df = pd.read_csv(path, **kwargs)
        return cls(data=df)

    @classmethod
    def load_df(cls: type[Self], df: pd.DataFrame) -> Self:
        return cls(data=df)

    # Plotting utilities
    def plot_diagnostics(
        self,
        show: bool = True,
        log_spend: bool = True,
    ) -> plt.Figure:
        """Create diagnostic plots to validate the simulated dataset.

        Plots:
        - Distributions of control/treatment spend (log10-transformed)
        - Distributions of control/treatment conversions
        - Scatter: spend vs conversions for control and treatment
        - Distributions of per-campaign CPA (spend / conversions) for control and treatment

        Parameters
        ----------
        show:
            If True, calls plt.show() before returning.

        Returns
        -------
        fig:
            Matplotlib Figure containing the subplots.
        """

        df = self.data.copy()

        # Opt-in to pandas future behavior to avoid downcasting warnings
        pd.set_option("future.no_silent_downcasting", True)

        # Computed columns for plotting
        eps = 0.1  # epsilon is quite big but we want to avoid division by zero or large ratio values
        # CPA = spend / conversions (guard against zero conversions)
        df["ctrl_cpa"] = df["ctrl_spend"] / np.clip(df["ctrl_convs"], eps, None)
        df["treat_cpa"] = df["treat_spend"] / np.clip(df["treat_convs"], eps, None)
        df["log10_ctrl_spend"] = np.log10(np.clip(df["ctrl_spend"], eps, None))
        df["log10_treat_spend"] = np.log10(np.clip(df["treat_spend"], eps, None))
        df["log10_ctrl_convs"] = np.log10(np.clip(df["ctrl_convs"], eps, None))
        df["log10_treat_convs"] = np.log10(np.clip(df["treat_convs"], eps, None))
        df["log10_ctrl_cpa"] = np.log10(np.clip(df["ctrl_cpa"], eps, None))
        df["log10_treat_cpa"] = np.log10(np.clip(df["treat_cpa"], eps, None))

        # Use seaborn defaults for consistent style
        sns.set_theme(style="whitegrid")

        # Create a single figure with a 3x2 grid; bottom row reserved for summary table
        fig = plt.figure(figsize=(16, 12))
        grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[1, 0])
        ax4 = fig.add_subplot(grid[1, 1])
        ax5 = fig.add_subplot(grid[2, :])

        # Spend distributions (overlay control and treatment)
        spend_col = "log10_ctrl_spend" if log_spend else "ctrl_spend"
        spend_treat_col = "log10_treat_spend" if log_spend else "treat_spend"
        spend_plot_df = pd.DataFrame(
            {
                "spend": pd.concat([df[spend_col], df[spend_treat_col]], ignore_index=True),
                "arm": (["control"] * len(df)) + (["treatment"] * len(df)),
            }
        )
        sns.histplot(
            data=spend_plot_df,
            x="spend",
            hue="arm",
            bins=40,
            alpha=0.5,
            palette={"control": "#1f77b4", "treatment": "#ff7f0e"},
            ax=ax1,
        )
        ax1.set_title("Spend" + (" (log10)" if log_spend else ""))
        ax1.set_xlabel("log10(spend)" if log_spend else "spend")
        ax1.set_ylabel("count")

        # Conversions distributions (log10-transformed)
        convs_plot_df = pd.DataFrame(
            {
                "conversions": pd.concat([df["log10_ctrl_convs"], df["log10_treat_convs"]], ignore_index=True),
                "arm": (["control"] * len(df)) + (["treatment"] * len(df)),
            }
        )
        sns.histplot(
            data=convs_plot_df,
            x="conversions",
            hue="arm",
            bins=40,
            alpha=0.5,
            palette={"control": "#1f77b4", "treatment": "#ff7f0e"},
            ax=ax2,
        )
        ax2.set_title("Conversions (log10)")
        ax2.set_xlabel("log10(conversions)")
        ax2.set_ylabel("count")

        # Spend vs conversions scatter
        scatter_df = pd.DataFrame(
            {
                "spend": pd.concat([df["ctrl_spend"], df["treat_spend"]], ignore_index=True),
                "conversions": pd.concat([df["ctrl_convs"], df["treat_convs"]], ignore_index=True),
                "arm": (["control"] * len(df)) + (["treatment"] * len(df)),
            }
        )
        sns.scatterplot(
            data=scatter_df,
            x="spend",
            y="conversions",
            hue="arm",
            palette={"control": "#1f77b4", "treatment": "#ff7f0e"},
            s=12,
            alpha=0.5,
            ax=ax3,
        )
        ax3.set_title("Spend vs Conversions")
        ax3.set_xlabel("spend")
        ax3.set_ylabel("conversions")

        fig.suptitle("SimulationData Diagnostics", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Additional subplot: CPA distributions overlay (log10-transformed)
        cpa_plot_df = pd.DataFrame(
            {
                "cpa": pd.concat([df["log10_ctrl_cpa"], df["log10_treat_cpa"]], ignore_index=True),
                "arm": (["control"] * len(df)) + (["treatment"] * len(df)),
            }
        )
        sns.histplot(
            data=cpa_plot_df,
            x="cpa",
            hue="arm",
            bins=50,
            alpha=0.5,
            palette={"control": "#1f77b4", "treatment": "#ff7f0e"},
            ax=ax4,
        )
        ax4.set_title("Per-Campaign CPA (log10)")
        ax4.set_xlabel("log10(CPA)")
        ax4.set_ylabel("count")

        # Summary table
        num_advertisers = int(pd.Series(df["adv_id"]).nunique())
        num_campaigns = int(pd.Series(df["cpg_id"]).nunique())
        total_ctrl_spend = float(pd.Series(df["ctrl_spend"]).sum())
        total_treat_spend = float(pd.Series(df["treat_spend"]).sum())
        total_ctrl_convs = int(pd.Series(df["ctrl_convs"]).sum())
        total_treat_convs = int(pd.Series(df["treat_convs"]).sum())
        agg_ctrl_cpa = total_ctrl_spend / max(total_ctrl_convs, 1)
        agg_treat_cpa = total_treat_spend / max(total_treat_convs, 1)

        table_rows = [
            ["Advertisers", f"{num_advertisers:d}"],
            ["Campaigns", f"{num_campaigns:d}"],
            ["Ctrl spend (sum)", f"{total_ctrl_spend:,.0f}"],
            ["Treat spend (sum)", f"{total_treat_spend:,.0f}"],
            ["Ctrl conversions (sum)", f"{total_ctrl_convs:,d}"],
            ["Treat conversions (sum)", f"{total_treat_convs:,d}"],
            ["Ctrl CPA (aggregate)", f"{agg_ctrl_cpa:,.2f}"],
            ["Treat CPA (aggregate)", f"{agg_treat_cpa:,.2f}"],
        ]
        ax5.axis("off")
        summary_table = ax5.table(cellText=table_rows, colLabels=["Metric", "Value"], loc="center")
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(10)
        summary_table.scale(1, 1.2)
        ax5.set_title("Dataset Summary", pad=10)

        if show:
            plt.show()

        return fig


@dataclass(frozen=True)
class ModelResult:
    """Output of a single statistical model fit on one dataset."""

    effect: float
    std_error: float
    interval_lower: float
    interval_upper: float
    p_value: float | None = None
    reject_h0: bool | None = None
    posterior_prob: float | None = None
    tau2: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated performance metrics over many simulation runs for one model."""

    coverage: float
    bias: float
    rmse: float
    mean_estimate: float
    std_estimate: float
    mean_interval_width: float
    power: float | None
    type_i_error: float | None
    mean_p_value: float | None
    mean_posterior_prob: float | None
    mean_tau2: float | None
    std_tau2: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
