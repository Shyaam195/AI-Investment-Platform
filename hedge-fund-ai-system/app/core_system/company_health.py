class HNIRanker:

    def classify(self, marketcap):
        if marketcap > 500_000:
            return ">500B"
        elif marketcap > 100_000:
            return "100B-500B"
        else:
            return "<100B"

    def rank(self, df):
        return df.sort_values("AI_Score", ascending=False)