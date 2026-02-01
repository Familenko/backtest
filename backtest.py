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
    available_sum = 10000,
    fee: float = 0.001,
    profit_multiple: float = 2.0,
    plot: bool = True
):
    buy_dates = prices.resample(freq).first().dropna().index

    qty = 0.0

    dates = []
    portfolio_history = []
    investment_history = []
    realized_profit = []
    invested_series = pd.Series(0.0, index=prices.index)
    realized_invested = 0.0
    profit = None
    trigger_dates = []

    # динамічний поріг для take-profit
    profit_multiplier_dynamic = profit_multiple

    counter = 0
    for date, price in prices.items():
        # --- купівля за зарплату ---
        if date in buy_dates:

            available_sum -= buy_amount
            if available_sum < 0:
                continue

            effective_amount = buy_amount * (1 - fee)
            qty += effective_amount / price
            invested_series.loc[date] += buy_amount

        invested_cum = invested_series.loc[:date].sum()
        net_invested = invested_cum - realized_invested
        net_portfolio_value = qty * price

        if counter != 0:
            counter -= 1

        # --- take-profit ---
        if (net_invested > 0) and (net_portfolio_value >= (profit_multiplier_dynamic * net_invested)) and counter == 0:
            # продаємо половину портфеля
            sell_qty = qty * 0.5
            proceeds = sell_qty * price * (1 - fee)
            cost_basis = net_invested * 0.5
            profit = proceeds - cost_basis

            qty -= sell_qty
            realized_invested += cost_basis
            trigger_dates.append(date)

            # збільшуємо поріг для наступного take-profit
            # profit_multiplier_dynamic *= 2
            counter = 180

            # оновлюємо net_portfolio_value після продажу
            net_portfolio_value = qty * price
            net_invested = invested_cum - realized_invested

        dates.append(date)
        portfolio_history.append(net_portfolio_value)
        investment_history.append(net_invested)
        if profit:
            realized_profit.append(profit)
            profit = None
        else:
            realized_profit.append(0.0)

    invested_cum = invested_series.cumsum()

    result = pd.DataFrame({
        "Portfolio": portfolio_history,
        "Invested": investment_history,
        "Realized_profit": realized_profit
    }, index=dates)

    metrics = {
        "Total_invested": invested_cum.iloc[-1],
        "Final_value": result["Portfolio"].iloc[-1],
        "Realized_profit": sum(realized_profit),
        "Num_rebalances": len(trigger_dates)
    }

    if plot:
        fig, ax_price = plt.subplots(figsize=(10, 5))

        ax_price.plot(prices.index, prices.values, color="orange", label="Asset price")
        ax_price.set_ylabel("Asset price ($)")
        ax_price.set_xlabel("Date")

        ax_portfolio = ax_price.twinx()
        ax_portfolio.plot(result.index, result["Portfolio"], label="Portfolio ($)")
        invested_plot = result["Invested"].resample("ME").last()
        ax_portfolio.bar(invested_plot.index, invested_plot.values, width=20, alpha=0.15, label="Invested capital ($)")

        ax_portfolio.set_ylabel("Portfolio / Invested ($)")

        l1, lab1 = ax_price.get_legend_handles_labels()
        l2, lab2 = ax_portfolio.get_legend_handles_labels()
        ax_price.legend(l1 + l2, lab1 + lab2, loc="upper left")

        plt.title(f"DCA ({freq}) — Price vs Portfolio vs Invested (Dynamic TP)")
        plt.show()

    return result, metrics

