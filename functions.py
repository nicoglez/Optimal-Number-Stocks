import numpy as np
import pandas as pd

class AssetAllocation:

    def __init__(self, data_stocks: pd.DataFrame, data_benchmark: pd.DataFrame, rf: float):
        self.stocks = data_stocks
        self.stocks_rends = data_stocks.pct_change().dropna()
        self.bench = data_benchmark
        self.bench_rends = data_benchmark.pct_change().dropna()
        self.rf = rf

    # Get metrics
    @staticmethod
    def get_metrics(w, mean, cov):
        returns = np.sum(mean * w) * 252
        var = np.dot(w, np.dot(cov, w)) ** 0.5 * 252
        std = np.dot(w, np.dot(cov, w)) ** 0.5 * (252 ** 0.5)
        return var, std, returns

    # Downside Risk
    @staticmethod
    def downside_risk(diff: pd.DataFrame):
        downside = diff[diff <= 0].fillna(0)
        std = downside.std()
        return np.array(std)

    # Sharpe Ratio optimization
    def sharpe_ratio(self, n_sims):
        mean = self.stocks_rends.mean()
        cov = self.stocks_rends.cov()
        n_stocks = len(self.stocks.columns)
        history = np.zeros((1, n_sims))
        w = []
        # Montecarlo simulation
        for i in range(n_sims):
            temp = np.random.uniform(0, 1, n_stocks)
            temp = temp / np.sum(temp)
            w.append(temp)
            var, std, r = self.get_metrics(temp, mean, cov)
            history[0, i] = (r - self.rf) / std

        # return optimal weights
        return w[np.argmax(history)]

    # Semivariance Optimization
    def semivariance(self, n_sims):
        rends=self.stocks_rends.copy()
        rends_b=self.bench_rends.copy()
        # Calculate downside risk
        diff = pd.DataFrame([rends.iloc[:, i]-rends_b for i in range(rends.shape[1])]).T
        std = self.downside_risk(diff)
        # Calculate semivar matrix
        semivar_matrix = np.multiply(std.reshape(len(std), 1), std) * diff.corr()
        # Montecarlo simulation
        mean = self.stocks_rends.mean()
        cov = self.stocks_rends.cov()
        n_stocks = len(self.stocks.columns)
        history = np.zeros((1, n_sims))
        w = []
        # Montecarlo simulation
        for i in range(n_sims):
            temp = np.random.uniform(0, 1, n_stocks)
            temp = temp / np.sum(temp)
            w.append(temp)
            history[0, i] = np.dot(temp, np.dot(semivar_matrix, temp))

        return w[np.argmin(history)]


