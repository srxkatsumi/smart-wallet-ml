import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date as _date, datetime
from pathlib import Path
from data.calendars import target_dates

logger = logging.getLogger(__name__)


def cleanup_old_charts(charts_dir: Path, days_to_keep: int = 30):
    pattern   = re.compile(r"_(\d{8})\.png$")
    hoje_d    = _date.today()
    cutoff    = (pd.Timestamp(hoje_d) - pd.offsets.BDay(days_to_keep)).date()
    removed   = kept = ignored = 0

    for ficheiro in charts_dir.glob("*.png"):
        match = pattern.search(ficheiro.name)
        if not match:
            ignored += 1
            continue
        ds = match.group(1)
        try:
            data_f = _date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
        except ValueError:
            ignored += 1
            continue
        if data_f < cutoff:
            ficheiro.unlink()
            removed += 1
        else:
            kept += 1

    logger.info("Limpeza de gráficos: corte=%s removidos=%d mantidos=%d ignorados=%d",
                cutoff.strftime("%d/%m/%Y"), removed, kept, ignored)


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a1a",
        "axes.edgecolor":   "#444",    "text.color":     "white",
        "xtick.color":      "#aaa",    "ytick.color":    "#aaa",
        "grid.color":       "#333",    "axes.labelcolor":"white",
    })


def _plot_single(ticker: str, res: dict, df_log: pd.DataFrame,
                 portfolio_config: dict, charts_dir: Path):
    nome = ticker

    df_plot   = res["df"].tail(120)
    close_now = res["close_now"]
    last_date = res["last_date"]

    fig = plt.figure(figsize=(14, 10), facecolor="#0f0f0f")
    gs  = gridspec.GridSpec(4, 1, height_ratios=[4, 1.5, 1.5, 1.5], hspace=0.08)
    fig.suptitle(f"{ticker} — {nome}", color="white", fontsize=13, fontweight="bold")

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_plot.index, df_plot["Close"],  color="#00bfff", linewidth=1.5, label="Preço")
    ax1.plot(df_plot.index, df_plot["SMA20"],  color="#ffa500", linewidth=1,   label="SMA20", alpha=0.8)
    ax1.plot(df_plot.index, df_plot["SMA50"],  color="#ff6b6b", linewidth=1,   label="SMA50", alpha=0.8)
    ax1.fill_between(df_plot.index, df_plot["BB_lower"], df_plot["BB_upper"],
                     alpha=0.06, color="cyan", label="Bollinger")

    pred_x = [last_date]
    pred_y = [close_now]
    tdates = target_dates(last_date, ticker)
    for day, (direction, pred_price, conf) in res["preds_dict"].items():
        fdate  = tdates.get(day, last_date + pd.offsets.BDay(day))
        color  = "#00ff88" if direction == "up" else "#ff4444"
        marker = "^" if direction == "up" else "v"
        ax1.scatter(fdate, pred_price, color=color, s=120, zorder=6, marker=marker,
                    label=f"D+{day}:{direction}({conf:.0%})")
        pred_x.append(fdate)
        pred_y.append(pred_price)
    ax1.plot(pred_x, pred_y, color="gray", linestyle=":", linewidth=1, alpha=0.4)

    # Apenas D+1: uma validação por dia útil, sem sobreposição de horizontes
    df_tick_log = df_log[df_log["ticker"] == ticker].copy()
    validated   = df_tick_log[
        (df_tick_log["correct"].notna()) &
        (df_tick_log["horizon"] == 1)
    ]
    if not validated.empty:
        corr = validated[validated["correct"] == True]
        err  = validated[validated["correct"] == False]
        if not corr.empty:
            ax1.scatter(pd.to_datetime(corr["target_date"]), corr["pred_price"],
                        marker="o", color="lime", s=60, alpha=0.8, label="D+1 correta")
        if not err.empty:
            ax1.scatter(pd.to_datetime(err["target_date"]), err["pred_price"],
                        marker="x", color="red", s=60, alpha=0.8, label="D+1 errada")

    ax1.set_ylabel("Preço", color="white")
    ax1.legend(fontsize=7, loc="upper left", ncol=4,
               facecolor="#1a1a1a", labelcolor="white", edgecolor="#444")
    ax1.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df_plot.index, df_plot["RSI14"], color="#cc44ff", linewidth=1)
    ax2.axhline(70, color="red",   linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.axhline(30, color="green", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.fill_between(df_plot.index, 70, df_plot["RSI14"].clip(lower=70), alpha=0.15, color="red")
    ax2.fill_between(df_plot.index, df_plot["RSI14"].clip(upper=30), 30, alpha=0.15, color="green")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="white")
    ax2.grid(True, alpha=0.2)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    colors_hist = ["#00ff88" if v >= 0 else "#ff4444" for v in df_plot["MACD_hist"]]
    ax3.bar(df_plot.index, df_plot["MACD_hist"], color=colors_hist, alpha=0.5, width=1)
    ax3.plot(df_plot.index, df_plot["MACD"],     color="#00bfff", linewidth=1, label="MACD")
    ax3.plot(df_plot.index, df_plot["MACD_sig"], color="orange",  linewidth=1, label="Signal")
    ax3.axhline(0, color="white", linewidth=0.5, alpha=0.5)
    ax3.set_ylabel("MACD", color="white")
    ax3.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", edgecolor="#444")
    ax3.grid(True, alpha=0.2)

    ax4 = fig.add_subplot(gs[3])
    if len(validated) >= 3:
        val_s          = validated.sort_values("target_date").copy()
        val_s["cum_acc"]= val_s["correct"].astype(float).expanding().mean() * 100
        ax4.plot(pd.to_datetime(val_s["target_date"]), val_s["cum_acc"],
                 color="teal", linewidth=2, label="Acurácia acumulada")
        ax4.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
                    label="Baseline 50%")
        ax4.set_ylim(0, 105)
        ax4.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white", edgecolor="#444")
        ax4.set_title("Acurácia das previsões", color="white", fontsize=9)
    else:
        ax4.text(0.5, 0.5, "Acurácia disponível após primeiras previsões validadas",
                 ha="center", va="center", transform=ax4.transAxes,
                 color="gray", fontsize=9)
    ax4.set_ylabel("Acurácia %", color="white")
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    safe      = ticker.replace(".", "_").replace("-", "_")
    data_hoje = datetime.now().strftime("%Y%m%d")
    path      = charts_dir / f"{safe}_{data_hoje}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)
    logger.info("%s → %s", ticker, path)


def generate_charts(my_tickers: list[str], resultados_ml: dict,
                    df_log: pd.DataFrame, portfolio_config: dict,
                    charts_dir: Path):
    _setup_style()
    for ticker in sorted(my_tickers):
        if ticker not in resultados_ml:
            continue
        try:
            _plot_single(ticker, resultados_ml[ticker], df_log,
                         portfolio_config, charts_dir)
        except Exception as e:
            logger.error("%s: erro no gráfico: %s", ticker, e)
    logger.info("Gráficos gerados para %d ativos",
                sum(1 for t in my_tickers if t in resultados_ml))
