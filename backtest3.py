import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def backtest_dca(
    prices: pd.Series,
    freq: str = "D",
    available_sum = 10000,
    fee: float = 0.001,
    profit_multiple: float = 2.0,
    plot: bool = True
):
    buy_dates = prices.resample(freq).first().dropna().index
    buy_amount = available_sum / len(buy_dates)

    qty = 0.0

    dates = []
    portfolio_history = []
    investment_history = []
    realized_profit = []
    buy_amount_list = []
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
        if (net_invested > 0) and (net_portfolio_value >= (profit_multiplier_dynamic * net_invested)):
            # продаємо половину портфеля
            sell_qty = qty * 0.5
            proceeds = sell_qty * price * (1 - fee)
            cost_basis = net_invested * 0.5
            profit = proceeds - cost_basis

            qty -= sell_qty
            realized_invested += cost_basis
            trigger_dates.append(date)

            # оновлюємо net_portfolio_value після продажу
            net_portfolio_value = qty * price
            net_invested = invested_cum - realized_invested

            available_sum = -1

        dates.append(date)
        portfolio_history.append(net_portfolio_value)
        investment_history.append(net_invested)
        if profit:
            realized_profit.append(profit)
            profit = None
        else:
            realized_profit.append(0.0)

        buy_amount_list.append(buy_amount)

    invested_cum = invested_series.cumsum()

    result = pd.DataFrame({
        "Portfolio": portfolio_history,
        "Invested": investment_history,
        "Realized_profit": realized_profit,
        "Buy_amount": buy_amount_list
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
        invested_plot = result["Invested"].resample("M").last()
        ax_portfolio.bar(invested_plot.index, invested_plot.values, width=20, alpha=0.15, label="Invested capital ($)")

        ax_portfolio.set_ylabel("Portfolio / Invested ($)")

        l1, lab1 = ax_price.get_legend_handles_labels()
        l2, lab2 = ax_portfolio.get_legend_handles_labels()
        ax_price.legend(l1 + l2, lab1 + lab2, loc="upper left")

        plt.title(f"DCA ({freq}) — Price vs Portfolio vs Invested (Dynamic TP)")
        plt.show()

    return result, metrics

