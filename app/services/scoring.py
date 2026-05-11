def calculate_score(growth, risk, rsi, volatility):
    growth = float(growth)
    risk = float(risk)
    rsi = float(rsi)
    volatility = float(volatility)

    return (
        growth * 0.4 +
        (100 - risk) * 0.3 +
        rsi * 0.2 +
        (100 - volatility) * 0.1
    )