import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def backtest_dca(
    target: str,
    prices: pd.Series,
    buy_amount: float = 2.0,
    freq: str = "D",
    available_sum: float = 1_000_000,
    fee: float = 0.001,
    profit_multiple: float = 2.0,
    cooldown_days: int = 180,
    plot: bool = True
):
    buy_dates = prices.resample(freq).first().dropna().index

    qty = 0.0
    cost_basis = 0.0
    realized_profit = 0.0
    returns_sum = 0.0
    cash_spent = 0.0

    dates = []
    portfolio_value = []
    invested_value = []
    realized_profit_series = []
    returns = []
    avg_price_series = []

    cooldown = 0
    trigger_dates = []

    for date, price in prices.items():

        # --- DCA покупка ---
        if date in buy_dates and available_sum >= buy_amount:
            available_sum -= buy_amount
            cash_spent += buy_amount

            effective_amount = buy_amount * (1 - fee)
            buy_qty = effective_amount / price

            qty += buy_qty
            cost_basis += buy_amount

        if cooldown > 0:
            cooldown -= 1

        # --- take-profit ---
        if (
            qty > 0
            and cost_basis > 0
            and qty * price >= profit_multiple * cost_basis
            and cooldown == 0
        ):
            sell_qty = qty * 0.5
            proceeds = sell_qty * price * (1 - fee)

            cost_sold = cost_basis * (sell_qty / qty)
            realized_profit += proceeds - cost_sold
            returns_sum += proceeds

            qty -= sell_qty
            cost_basis -= cost_sold

            cooldown = cooldown_days
            trigger_dates.append(date)

        avg_price = cost_basis / qty if qty > 0 else np.nan

        dates.append(date)
        portfolio_value.append(qty * price)
        invested_value.append(cost_basis)
        realized_profit_series.append(realized_profit)
        returns.append(returns_sum)
        avg_price_series.append(avg_price)

    result = pd.DataFrame(
        {
            "Portfolio": portfolio_value,
            "Invested": invested_value,
            "Price": prices.values,
            "Avg_price": avg_price_series,
            "Realized_profit": realized_profit_series,
            "Returns": returns,
        },
        index=dates,
    )

    result['Averege_dominance'] = result['Price'] >= result['Avg_price']
    bull_history = result['Averege_dominance'].value_counts(normalize=True).iloc[0]
    bull_history = int(bull_history * 100)

    metrics = {
        "Cash_spent": int(cash_spent),
        "Final_portfolio_value": int(portfolio_value[-1]),
        "Realized_profit": int(realized_profit),
        "Total_returns": int(returns_sum),
        "Total_equity": int(portfolio_value[-1]) + int(returns_sum),
        "Num_take_profits": len(trigger_dates),
        "Bull_history": bull_history
    }

    if plot:
        plot_graph(target, metrics, result, trigger_dates)

    return result, metrics


def plot_graph(target, metrics, result, trigger_dates):
    fig, ax_price = plt.subplots(figsize=(10, 5))

    # --- ціна активу ---
    ax_price.plot(
        result.index,
        result["Price"],
        color="orange",
        label="Asset price",
        linewidth=1.0,
    )

    # --- середня ціна ---
    ax_price.plot(
        result.index,
        result["Avg_price"],
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label="Average price",
    )

    ax_price.set_ylabel("Asset price ($)")
    ax_price.set_xlabel("Date")

    # --- тейк-профіти ---
    if trigger_dates:
        ax_price.scatter(
            trigger_dates,
            result.loc[trigger_dates]["Price"],
            color="red",
            marker="o",
            s=20,
            zorder=5,
            label="Take-profit",
        )

    for dt in trigger_dates:
        profit = result.loc[dt]["Realized_profit"]
        price = result.loc[dt]["Price"]

        ax_price.annotate(
            f"{profit:.0f}$",
            xy=(dt, price),
            xytext=(0, 8),
            textcoords="offset points",
            ha="right",
            fontsize=12,
            color="black",
        )

    # --- портфель ---
    ax_portfolio = ax_price.twinx()
    ax_portfolio.plot(
        result.index,
        result["Portfolio"],
        label="Portfolio ($)",
        alpha=0.5,
    )

    # --- вкладення ---
    invested_plot = result["Invested"].resample("ME").last()
    ax_portfolio.bar(
        invested_plot.index,
        invested_plot.values,
        width=20,
        alpha=0.1,
        label="Invested capital ($)",
    )

    ax_portfolio.set_ylabel("Portfolio / Invested ($)")

    # --- легенда ---
    l1, lab1 = ax_price.get_legend_handles_labels()
    l2, lab2 = ax_portfolio.get_legend_handles_labels()
    ax_price.legend(l1 + l2, lab1 + lab2, loc="upper left")

    plt.title(f"<{target}> Cash_spent: {metrics['Cash_spent']}, Final_portfolio_value: {metrics['Final_portfolio_value']}, Bull_history: {metrics['Bull_history']}%," +
                f"\n Realized_profit: {metrics['Realized_profit']}, Total_returns: {metrics['Total_returns']}, Total_equity: {metrics['Total_equity']}, Num_take_profits: {metrics['Num_take_profits']}")
    
    plt.tight_layout()
    plt.show()
