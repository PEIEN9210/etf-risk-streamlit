# -*- coding: utf-8 -*-
"""
ETF 推薦系統回測引擎
====================

功能：
1. 歷史回測（2015-2024）
2. 多策略比較
3. 統計顯著性檢驗
4. 風險指標計算
5. 視覺化分析

理論依據：
- DeMiguel et al. (2009): 1/N策略難以擊敗
- Harvey et al. (2016): 回測過度擬合問題
- Bailey et al. (2014): 偽發現機率控制
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===============================
# 常數設定
# ===============================
TRADING_DAYS = 252
RISK_FREE_RATE = 0.015
TRANSACTION_COST = 0.001425  # 0.1425%

ETF_UNIVERSE = {
    "0050.TW": "股票型",
    "006208.TW": "股票型",
    "00692.TW": "股票型",
    "00757.TW": "股票型",
    "0056.TW": "高股息型",
    "00878.TW": "高股息型",
    "00919.TW": "高股息型",
}

# ===============================
# 資料結構
# ===============================
@dataclass
class Portfolio:
    """投資組合"""
    name: str
    initial_capital: float = 1_000_000
    holdings: Dict[str, float] = field(default_factory=dict)  # {ETF: 股數}
    cash: float = 0.0
    history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.cash = self.initial_capital
    
    def get_value(self, prices: pd.Series) -> float:
        """計算投資組合市值"""
        holdings_value = sum(
            shares * prices.get(etf, 0) 
            for etf, shares in self.holdings.items()
        )
        return holdings_value + self.cash
    
    def rebalance(self, target_weights: Dict[str, float], 
                  prices: pd.Series, date: datetime) -> Dict[str, float]:
        """重新配置"""
        total_value = self.get_value(prices)
        
        # 計算目標持股
        target_holdings = {}
        for etf, weight in target_weights.items():
            target_value = total_value * weight
            target_shares = target_value / prices[etf]
            target_holdings[etf] = target_shares
        
        # 計算交易
        trades = {}
        for etf in set(list(self.holdings.keys()) + list(target_holdings.keys())):
            current = self.holdings.get(etf, 0)
            target = target_holdings.get(etf, 0)
            trades[etf] = target - current
        
        # 執行交易
        transaction_costs = 0
        for etf, delta_shares in trades.items():
            if abs(delta_shares) < 0.01:  # 忽略微小交易
                continue
            
            trade_value = abs(delta_shares * prices[etf])
            cost = trade_value * TRANSACTION_COST
            transaction_costs += cost
            
            self.holdings[etf] = self.holdings.get(etf, 0) + delta_shares
        
        # 更新現金
        self.cash = total_value - sum(
            shares * prices[etf] 
            for etf, shares in self.holdings.items()
        ) - transaction_costs
        
        # 記錄
        self.history.append({
            'date': date,
            'value': self.get_value(prices),
            'holdings': self.holdings.copy(),
            'cash': self.cash,
            'transaction_cost': transaction_costs
        })
        
        return trades

@dataclass
class BacktestResult:
    """回測結果"""
    portfolio_name: str
    portfolio_values: pd.Series
    returns: pd.Series
    holdings_history: pd.DataFrame
    
    # 績效指標
    total_return: float = 0.0
    ann_return: float = 0.0
    ann_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # 風險指標
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # 交易統計
    total_trades: int = 0
    total_costs: float = 0.0
    turnover: float = 0.0
    
    def calculate_metrics(self):
        """計算所有指標"""
        # 報酬指標
        self.total_return = (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1
        
        n_years = len(self.returns) / TRADING_DAYS
        self.ann_return = (1 + self.total_return) ** (1 / n_years) - 1
        self.ann_volatility = self.returns.std() * np.sqrt(TRADING_DAYS)
        
        # Sharpe Ratio
        excess_returns = self.returns - RISK_FREE_RATE / TRADING_DAYS
        self.sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS)
        
        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS)
        self.sortino_ratio = (self.ann_return - RISK_FREE_RATE) / downside_std if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        self.max_drawdown = drawdown.min()
        
        # Calmar Ratio
        self.calmar_ratio = self.ann_return / abs(self.max_drawdown) if self.max_drawdown != 0 else 0
        
        # VaR & CVaR
        self.var_95 = self.returns.quantile(0.05)
        self.cvar_95 = self.returns[self.returns <= self.var_95].mean()

# ===============================
# 策略定義
# ===============================
class Strategy:
    """基礎策略類別"""
    
    def __init__(self, name: str):
        self.name = name
    
    def select_etfs(self, data: Dict[str, pd.DataFrame], 
                   date: datetime, 
                   lookback_months: int = 12) -> Dict[str, float]:
        """選擇ETF與權重"""
        raise NotImplementedError

class UtilityBasedStrategy(Strategy):
    """效用函數策略"""
    
    def __init__(self, name: str, risk_aversion: float, 
                 dividend_preference: float = 0.5, top_n: int = 3):
        super().__init__(name)
        self.gamma = risk_aversion
        self.dividend_pref = dividend_preference
        self.top_n = top_n
    
    def select_etfs(self, data: Dict[str, pd.DataFrame], 
                   market_data: pd.DataFrame,
                   date: datetime, 
                   lookback_months: int = 12) -> Dict[str, float]:
        """基於效用函數選擇"""
        lookback_days = lookback_months * 21
        
        utilities = {}
        
        for etf, df in data.items():
            if df is None or len(df) < lookback_days:
                continue
            
            # 取得lookback期間資料
            end_idx = df.index.get_loc(date, method='ffill')
            start_idx = max(0, end_idx - lookback_days)
            
            etf_prices = df.iloc[start_idx:end_idx]['Close']
            market_prices = market_data.iloc[start_idx:end_idx]['Close']
            
            if len(etf_prices) < 50:
                continue
            
            # 計算報酬
            returns = etf_prices.pct_change().dropna()
            ann_return = returns.mean() * TRADING_DAYS
            ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
            
            # 簡化配息（實際應使用真實配息資料）
            dividend_yield = 0.03 if '0056' in etf or '00878' in etf or '00919' in etf else 0.02
            
            # 效用計算
            total_return = ann_return + dividend_yield * (1 + self.dividend_pref)
            risk_penalty = (self.gamma / 2) * (ann_vol ** 2)
            utility = total_return - risk_penalty
            
            utilities[etf] = utility
        
        # 選擇Top N
        sorted_etfs = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_etfs[:self.top_n]
        
        # 等權重
        weights = {etf: 1.0 / self.top_n for etf, _ in selected}
        
        return weights

class SharpeBasedStrategy(Strategy):
    """Sharpe Ratio策略"""
    
    def __init__(self, name: str, top_n: int = 3):
        super().__init__(name)
        self.top_n = top_n
    
    def select_etfs(self, data: Dict[str, pd.DataFrame], 
                   market_data: pd.DataFrame,
                   date: datetime, 
                   lookback_months: int = 12) -> Dict[str, float]:
        """基於Sharpe Ratio選擇"""
        lookback_days = lookback_months * 21
        
        sharpes = {}
        
        for etf, df in data.items():
            if df is None or len(df) < lookback_days:
                continue
            
            end_idx = df.index.get_loc(date, method='ffill')
            start_idx = max(0, end_idx - lookback_days)
            
            returns = df.iloc[start_idx:end_idx]['Close'].pct_change().dropna()
            
            if len(returns) < 50:
                continue
            
            ann_return = returns.mean() * TRADING_DAYS
            ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
            
            sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
            sharpes[etf] = sharpe
        
        sorted_etfs = sorted(sharpes.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_etfs[:self.top_n]
        
        weights = {etf: 1.0 / self.top_n for etf, _ in selected}
        
        return weights

class EqualWeightStrategy(Strategy):
    """等權重策略"""
    
    def __init__(self, name: str, etfs: List[str]):
        super().__init__(name)
        self.etfs = etfs
    
    def select_etfs(self, data: Dict[str, pd.DataFrame], 
                   market_data: pd.DataFrame,
                   date: datetime, 
                   lookback_months: int = 12) -> Dict[str, float]:
        """固定等權重"""
        weight = 1.0 / len(self.etfs)
        return {etf: weight for etf in self.etfs}

# ===============================
# 回測引擎
# ===============================
class BacktestEngine:
    """回測引擎"""
    
    def __init__(self, start_date: str, end_date: str, 
                 rebalance_freq: str = 'Q', lookback_months: int = 12):
        """
        參數:
        - start_date: 回測起始日 (YYYY-MM-DD)
        - end_date: 回測結束日 (YYYY-MM-DD)
        - rebalance_freq: 重新平衡頻率 ('M'=月, 'Q'=季, 'S'=半年, 'A'=年)
        - lookback_months: 估計參數的回顧期間（月）
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.rebalance_freq = rebalance_freq
        self.lookback_months = lookback_months
        
        # 資料準備期間（需要額外的lookback期間）
        self.data_start_date = self.start_date - timedelta(days=lookback_months * 30 + 30)
        
        self.price_data = {}
        self.market_data = None
    
    def fetch_data(self):
        """抓取歷史資料"""
        logger.info(f"抓取資料：{self.data_start_date.date()} 至 {self.end_date.date()}")
        
        all_tickers = list(ETF_UNIVERSE.keys()) + ["0050.TW"]
        
        for ticker in all_tickers:
            try:
                logger.info(f"下載 {ticker}...")
                df = yf.download(ticker, start=self.data_start_date, end=self.end_date, 
                               progress=False)
                
                if not df.empty and len(df) >= 50:
                    self.price_data[ticker] = df
                    logger.info(f"✓ {ticker}: {len(df)} 筆資料")
                else:
                    logger.warning(f"✗ {ticker}: 資料不足")
                    
            except Exception as e:
                logger.error(f"✗ {ticker}: {str(e)}")
        
        # 市場基準
        self.market_data = self.price_data.get("0050.TW")
        
        logger.info(f"資料抓取完成：{len(self.price_data)} 檔ETF")
    
    def get_rebalance_dates(self) -> List[datetime]:
        """產生重新平衡日期"""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq=self.rebalance_freq)
        
        # 確保日期在交易日
        valid_dates = []
        for date in dates:
            if self.market_data is not None:
                try:
                    idx = self.market_data.index.get_loc(date, method='ffill')
                    actual_date = self.market_data.index[idx]
                    valid_dates.append(actual_date)
                except:
                    continue
        
        return valid_dates
    
    def run_backtest(self, strategy: Strategy, 
                    initial_capital: float = 1_000_000) -> BacktestResult:
        """執行單一策略回測"""
        logger.info(f"回測策略: {strategy.name}")
        
        portfolio = Portfolio(name=strategy.name, initial_capital=initial_capital)
        rebalance_dates = self.get_rebalance_dates()
        
        logger.info(f"重新平衡日期: {len(rebalance_dates)} 次")
        
        for i, date in enumerate(rebalance_dates):
            logger.info(f"[{i+1}/{len(rebalance_dates)}] {date.date()}")
            
            # 選擇ETF
            try:
                weights = strategy.select_etfs(
                    self.price_data, 
                    self.market_data,
                    date, 
                    self.lookback_months
                )
            except Exception as e:
                logger.error(f"策略選擇失敗: {str(e)}")
                continue
            
            if not weights:
                logger.warning("未選擇任何ETF，跳過")
                continue
            
            # 取得當日價格
            prices = pd.Series({
                etf: self.price_data[etf].loc[date, 'Close'] 
                for etf in weights.keys()
            })
            
            # 重新平衡
            portfolio.rebalance(weights, prices, date)
        
        # 計算每日淨值
        all_dates = self.market_data.loc[self.start_date:self.end_date].index
        portfolio_values = []
        
        for date in all_dates:
            prices = pd.Series({
                etf: self.price_data[etf].loc[date, 'Close'] 
                for etf in portfolio.holdings.keys()
            })
            value = portfolio.get_value(prices)
            portfolio_values.append(value)
        
        portfolio_values = pd.Series(portfolio_values, index=all_dates)
        returns = portfolio_values.pct_change().dropna()
        
        # 整理持倉歷史
        holdings_df = pd.DataFrame([
            {'date': h['date'], **h['holdings']} 
            for h in portfolio.history
        ])
        
        # 計算交易統計
        total_costs = sum(h['transaction_cost'] for h in portfolio.history)
        
        # 建立結果
        result = BacktestResult(
            portfolio_name=strategy.name,
            portfolio_values=portfolio_values,
            returns=returns,
            holdings_history=holdings_df,
            total_costs=total_costs,
            total_trades=len(portfolio.history)
        )
        
        result.calculate_metrics()
        
        logger.info(f"✓ 完成: 總報酬 {result.total_return*100:.2f}%, Sharpe {result.sharpe_ratio:.3f}")
        
        return result
    
    def run_multiple_backtests(self, strategies: List[Strategy]) -> Dict[str, BacktestResult]:
        """執行多策略回測"""
        results = {}
        
        for strategy in strategies:
            result = self.run_backtest(strategy)
            results[strategy.name] = result
        
        return results

# ===============================
# 統計檢驗
# ===============================
class StatisticalTests:
    """統計檢驗工具"""
    
    @staticmethod
    def sharpe_difference_test(returns1: pd.Series, returns2: pd.Series) -> Dict:
        """
        Jobson & Korkie (1981) Sharpe Ratio差異檢驗
        H0: Sharpe1 = Sharpe2
        """
        n = min(len(returns1), len(returns2))
        
        # 計算Sharpe
        sharpe1 = (returns1.mean() - RISK_FREE_RATE/TRADING_DAYS) / returns1.std() * np.sqrt(TRADING_DAYS)
        sharpe2 = (returns2.mean() - RISK_FREE_RATE/TRADING_DAYS) / returns2.std() * np.sqrt(TRADING_DAYS)
        
        # 協方差
        returns_aligned = pd.DataFrame({
            'r1': returns1.iloc[:n],
            'r2': returns2.iloc[:n]
        }).dropna()
        
        cov_matrix = returns_aligned.cov()
        
        # 檢驗統計量
        var_diff = (1 + 0.5 * sharpe1**2) / n + (1 + 0.5 * sharpe2**2) / n - \
                   2 * cov_matrix.iloc[0,1] / (returns1.std() * returns2.std() * n)
        
        if var_diff <= 0:
            return {'difference': sharpe1 - sharpe2, 'p_value': np.nan, 'significant': False}
        
        z_stat = (sharpe1 - sharpe2) / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'sharpe1': sharpe1,
            'sharpe2': sharpe2,
            'difference': sharpe1 - sharpe2,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def bootstrap_confidence_interval(returns: pd.Series, 
                                     metric: str = 'sharpe',
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Bootstrap信賴區間"""
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            sample = returns.sample(n=len(returns), replace=True)
            
            if metric == 'sharpe':
                value = (sample.mean() - RISK_FREE_RATE/TRADING_DAYS) / sample.std() * np.sqrt(TRADING_DAYS)
            elif metric == 'return':
                value = sample.mean() * TRADING_DAYS
            elif metric == 'volatility':
                value = sample.std() * np.sqrt(TRADING_DAYS)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            bootstrap_metrics.append(value)
        
        lower = np.percentile(bootstrap_metrics, (1 - confidence_level) / 2 * 100)
        upper = np.percentile(bootstrap_metrics, (1 + confidence_level) / 2 * 100)
        
        return lower, upper

# ===============================
# 視覺化
# ===============================
class BacktestVisualizer:
    """回測視覺化"""
    
    @staticmethod
    def plot_cumulative_returns(results: Dict[str, BacktestResult], 
                               figsize=(14, 8), save_path=None):
        """累積報酬曲線"""
        plt.figure(figsize=figsize)
        
        for name, result in results.items():
            cumulative = (1 + result.returns).cumprod()
            plt.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
        
        plt.title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_drawdowns(results: Dict[str, BacktestResult], 
                      figsize=(14, 8), save_path=None):
        """回撤分析"""
        fig, axes = plt.subplots(len(results), 1, figsize=figsize, sharex=True)
        
        if len(results) == 1:
            axes = [axes]
        
        for ax, (name, result) in zip(axes, results.items()):
            cumulative = (1 + result.returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
            ax.set_title(f'{name} - Max DD: {result.max_drawdown*100:.2f}%', fontsize=12)
            ax.set_ylabel('Drawdown', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.xlabel('Date', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_performance_table(results: Dict[str, BacktestResult]):
        """績效指標表格"""
        metrics_data = []
        
        for name, result in results.items():
            metrics_data.append({
                'Strategy': name,
                'Total Return': f"{result.total_return*100:.2f}%",
                'Ann. Return': f"{result.ann_return*100:.2f}%",
                'Ann. Vol': f"{result.ann_volatility*100:.2f}%",
                'Sharpe': f"{result.sharpe_ratio:.3f}",
                'Sortino': f"{result.sortino_ratio:.3f}",
                'Max DD': f"{result.max_drawdown*100:.2f}%",
                'Calmar': f"{result.calmar_ratio:.3f}",
                'Total Costs': f"${result.total_costs:,.0f}"
            })
        
        df = pd.DataFrame(metrics_data)
        
        print("\n" + "="*100)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")
        
        return df

# ===============================
# 主執行腳本
# ===============================
if __name__ == "__main__":
    
    print("="*80)
    print("ETF 推薦系統回測引擎")
    print("="*80)
    
    # 1. 初始化回測引擎
    engine = BacktestEngine(
        start_date="2020-01-01",  # 先用4年測試，確認無誤後再擴展到2015
        end_date="2024-12-31",
        rebalance_freq='Q',  # 季度重新平衡
        lookback_months=12
    )
    
    # 2. 抓取資料
    engine.fetch_data()
    
    # 3. 定義策略
    strategies = [
        # 個人化策略（不同風險偏好）
        UtilityBasedStrategy("保守型 (γ=4)", risk_aversion=4.0, dividend_preference=0.8, top_n=3),
        UtilityBasedStrategy("穩健型 (γ=2.5)", risk_aversion=2.5, dividend_preference=0.5, top_n=3),
        UtilityBasedStrategy("積極型 (γ=1)", risk_aversion=1.0, dividend_preference=0.2, top_n=3),
        
        # 基準策略
        SharpeBasedStrategy("Sharpe排序", top_n=3),
        EqualWeightStrategy("等權0050+0056", etfs=["0050.TW", "0056.TW"]),
    ]
    
    # 4. 執行回測
    print("\n開始回測...")
    results = engine.run_multiple_backtests(strategies)
    
    # 5. 績效分析
    print("\n" + "="*80)
    print("績效分析")
    print("="*80)
    
    visualizer = BacktestVisualizer()
    performance_df = visualizer.plot_performance_table(results)
    
    # 6. 統計檢驗
    print("\n" + "="*80)
    print("統計顯著性檢驗")
    print("="*80)
    
    tester = StatisticalTests()
    
    # 比較保守型 vs 等權
    test_result = tester.sharpe_difference_test(
        results["保守型 (γ=4)"].returns,
        results["等權0050+0056"].returns
    )
    
    print(f"\n保守型 vs 等權0050+0056:")
    print(f"  Sharpe1: {test_result['sharpe1']:.3f}")
    print(f"  Sharpe2: {test_result['sharpe2']:.3f}")
    print(f"  差異: {test_result['difference']:.3f}")
    print(f"  p-value: {test_result['p_value']:.4f}")
    print(f"  顯著?: {'是' if test_result['significant'] else '否'}")
    
    # Bootstrap信賴區間
    lower, upper = tester.bootstrap_confidence_interval(
        results["保守型 (γ=4)"].returns, 
        metric='sharpe',
        n_bootstrap=1000
    )
    
    print(f"\n保守型 Sharpe Ratio 95% 信賴區間: [{lower:.3f}, {upper:.3f}]")
    
    # 7. 視覺化
    print("\n生成視覺化圖表...")
    
    visualizer.plot_cumulative_returns(results, save_path='cumulative_returns.png')
    visualizer.plot_drawdowns(results, save_path='drawdowns.png')
    
    print("\n✓ 回測完成！")
    print("  - 績效表格已顯示")
    print("  - 圖表已儲存: cumulative_returns.png, drawdowns.png")
