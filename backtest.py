import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def backtest_rebalance(df, 
                       target_weights=None, 
                       initial_capital=1.0, 
                       threshold=0.10, 
                       fee=0.001, 
                       plot=True):
    """
    Універсальний backtest ребалансування портфеля.

    Параметри:
    -----------
    df : pd.DataFrame
        Датафрейм з цінами активів. Колонки = назви активів, індекс = дата.
    target_weights : list або np.array
        Цільові ваги активів (сума = 1). Якщо None, рівномірний розподіл.
    initial_capital : float
        Початковий капітал.
    threshold : float
        Поріг ребалансу (10% = 0.10)
    fee : float
        Комісія на оборот портфеля (як частка, наприклад 0.001 = 0.1%)
    plot : bool
        Малювати графік чи ні.
    
    Повертає:
    ---------
    result : pd.DataFrame
        Портфель Rebalance та HODL для кожного активу.
    metrics : dict
        Основні метрики фінального портфеля.
    """

    assets = df.columns
    n_assets = len(assets)
    
    # Якщо ваги не задані, робимо рівномірний розподіл
    if target_weights is None:
        target_weights = np.array([1/n_assets]*n_assets)
    else:
        target_weights = np.array(target_weights)
        assert len(target_weights) == n_assets, "Кількість ваг повинна дорівнювати кількості активів"
        assert np.isclose(target_weights.sum(), 1.0), "Сума ваг повинна бути 1"

    # ----------------------------
    # Початковий портфель у кількості монет
    # ----------------------------
    p0 = df.iloc[0]
    qty = np.array([(initial_capital * w) / p0[a] for w, a in zip(target_weights, assets)])

    history = []
    rebalance_dates = []

    # ----------------------------
    # Основний цикл
    # ----------------------------
    for date, row in df.iterrows():
        values = qty * row.values
        total_value = values.sum()
        current_weights = values / total_value

        if np.any(np.abs(current_weights - target_weights) > threshold):
            # цільові вартості
            target_values = total_value * target_weights
            target_qty = target_values / row.values

            turnover = np.abs((target_qty - qty) * row.values).sum()
            cost = turnover * fee
            total_value -= cost

            # оновлюємо кількість монет
            qty = (total_value * target_weights) / row.values
            rebalance_dates.append(date)

        history.append(total_value)

    history = pd.Series(history, index=df.index)

    # ----------------------------
    # HODL для кожного активу
    # ----------------------------
    hodl = pd.DataFrame({f"HODL_{a}": df[a] / df[a].iloc[0] for a in assets})

    result = pd.DataFrame({"Rebalance": history / history.iloc[0]})
    result = pd.concat([result, hodl], axis=1)

    # ----------------------------
    # Метрика max drawdown
    # ----------------------------
    def max_drawdown(series):
        cum_max = series.cummax()
        drawdown = (series - cum_max) / cum_max
        return drawdown.min()

    metrics = {
        "Rebalance_final": result["Rebalance"].iloc[-1],
        "Rebalance_MDD": max_drawdown(result["Rebalance"]),
        "Num_rebalances": len(rebalance_dates)
    }

    for a in assets:
        metrics[f"HODL_{a}_final"] = result[f"HODL_{a}"].iloc[-1]
        metrics[f"HODL_{a}_MDD"] = max_drawdown(result[f"HODL_{a}"])

    # ----------------------------
    # Графік
    # ----------------------------
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(result["Rebalance"], label="Rebalance")
        for a in assets:
            plt.plot(result[f"HODL_{a}"], label=f"HODL {a}")
        plt.legend()
        plt.title("Backtest Rebalance Portfolio")
        plt.show()

    return result, metrics


def backtest_dca(
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
            "Avg_price": avg_price_series,
            "Realized_profit": realized_profit_series,
            "Returns": returns,
        },
        index=dates,
    )

    metrics = {
        "Cash_spent": int(cash_spent),
        "Final_portfolio_value": int(portfolio_value[-1]),
        "Realized_profit": int(realized_profit),
        "Total_returns": int(returns_sum),
        "Total_equity": int(portfolio_value[-1]) + int(returns_sum),
        "Num_take_profits": len(trigger_dates),
    }

    if plot:
        fig, ax_price = plt.subplots(figsize=(10, 5))

        # --- ціна активу ---
        ax_price.plot(
            prices.index,
            prices.values,
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
                prices.loc[trigger_dates],
                color="red",
                marker="o",
                s=20,
                zorder=5,
                label="Take-profit",
            )

        for dt in trigger_dates:
            profit = result.loc[dt]["Realized_profit"]
            price = prices.loc[dt]

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

        plt.title(f"DCA ({freq}) — Price vs Avg vs Portfolio (Take-Profit)")
        plt.tight_layout()
        plt.show()

    return result, metrics
