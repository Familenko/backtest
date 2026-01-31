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
    freq: str = "D",          # "D", "W", "M"
    fee: float = 0.001,
    plot: bool = True
):
    """
    Backtest Dollar-Cost Averaging (DCA).

    parameters
    ----------
    prices : pd.Series
        Ціни активу (index = дата, values = ціна)
    buy_amount : float
        Сума покупки в доларах за період
    freq : str
        Частота покупки: D / W / M
    fee : float
        Комісія (0.001 = 0.1%)
    plot : bool
        Малювати графік чи ні

    returns
    -------
    result : pd.DataFrame
    metrics : dict
    """

    # Дати покупок
    buy_dates = prices.resample(freq).first().dropna()

    qty = 0.0
    invested = 0.0

    history = []

    for date, price in prices.items():
        if date in buy_dates.index:
            effective_amount = buy_amount * (1 - fee)
            qty += effective_amount / price
            invested += buy_amount

        portfolio_value = qty * price
        history.append(portfolio_value)

    result = pd.DataFrame({
        "Portfolio": history,
        "Invested": invested
    }, index=prices.index)

    result["Normalized"] = result["Portfolio"] / result["Invested"]

    def max_drawdown(series):
        peak = series.cummax()
        dd = (series - peak) / peak
        return dd.min()

    metrics = {
        "Total_invested": invested,
        "Final_value": result["Portfolio"].iloc[-1],
        "ROI": result["Portfolio"].iloc[-1] / invested - 1,
        "MDD": max_drawdown(result["Portfolio"]),
        "Total_qty": qty
    }

    if plot:
        fig, ax_price = plt.subplots(figsize=(10, 5))

        # --- Ліва вісь: ціна активу ---
        ax_price.plot(
            prices.index,
            prices.values,
            label="SOL price",
            color="orange"
        )
        ax_price.set_ylabel("Asset price ($)")
        ax_price.set_xlabel("Date")

        # --- Права вісь: портфель + інвестована сума ---
        ax_portfolio = ax_price.twinx()

        # Лінія портфеля
        ax_portfolio.plot(
            result.index,
            result["Portfolio"],
            label="DCA portfolio ($)"
        )

        # --- Bar інвестованої суми (кумулятивно) ---
        invested_series = (
            prices
            .resample("D")
            .first()
            .notna()
            .cumsum() * buy_amount
        )

        # Щоб бари не були щоденним шумом — агрегуємо по місяцях
        invested_monthly = invested_series.resample("M").last()

        ax_portfolio.bar(
            invested_monthly.index,
            invested_monthly.values,
            width=20,
            alpha=0.15,
            label="Invested capital ($)"
        )

        ax_portfolio.set_ylabel("Portfolio / Invested ($)")

        # --- Легенда ---
        lines_1, labels_1 = ax_price.get_legend_handles_labels()
        lines_2, labels_2 = ax_portfolio.get_legend_handles_labels()
        ax_price.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="upper left"
        )

        plt.title("DCA: Asset Price vs Portfolio vs Invested Capital")
        plt.show()


    return result, metrics
