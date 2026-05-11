class MultiAgentSystem:

    def __init__(self, risk_agent, market_agent, health_agent):
        self.risk_agent = risk_agent
        self.market_agent = market_agent
        self.health_agent = health_agent

    def analyze(self, data):
        return {
            "risk": self.risk_agent(data["Risk"], data["Volatility"]),
            "trend": self.market_agent(data["RSI"]),
            "health": self.health_agent(data["Growth"], data["Risk"])
        }