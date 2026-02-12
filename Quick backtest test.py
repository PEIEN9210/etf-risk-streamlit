#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速回測測試腳本
===============
用途：快速驗證回測引擎是否正常運作
時間：約2-3分鐘
"""

import sys
import pandas as pd
from backtest_engine import (
    BacktestEngine, 
    UtilityBasedStrategy, 
    SharpeBasedStrategy,
    EqualWeightStrategy,
    BacktestVisualizer,
    StatisticalTests
)

def quick_test():
    """快速測試（2年資料）"""
    
    print("="*80)
    print("快速回測測試（2年資料）")
    print("="*80)
    
    # 1. 初始化（只測試2年）
    engine = BacktestEngine(
        start_date="2023-01-01",
        end_date="2024-12-31",
        rebalance_freq='Q',
        lookback_months=12
    )
    
    # 2. 抓取資料
    print("\n步驟1: 抓取資料...")
    engine.fetch_data()
    
    if engine.market_data is None:
        print("✗ 市場資料抓取失敗")
        return
    
    print(f"✓ 成功抓取 {len(engine.price_data)} 檔ETF資料")
    
    # 3. 定義測試策略
    print("\n步驟2: 定義策略...")
    strategies = [
        UtilityBasedStrategy("保守型", risk_aversion=4.0, top_n=3),
        UtilityBasedStrategy("積極型", risk_aversion=1.0, top_n=3),
        EqualWeightStrategy("等權基準", etfs=["0050.TW", "0056.TW"]),
    ]
    print(f"✓ 已定義 {len(strategies)} 個策略")
    
    # 4. 執行回測
    print("\n步驟3: 執行回測...")
    try:
        results = engine.run_multiple_backtests(strategies)
        print(f"✓ 回測完成")
    except Exception as e:
        print(f"✗ 回測失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 顯示結果
    print("\n步驟4: 績效分析...")
    visualizer = BacktestVisualizer()
    performance_df = visualizer.plot_performance_table(results)
    
    # 6. 簡單檢驗
    print("\n步驟5: 統計檢驗...")
    tester = StatisticalTests()
    
    if "保守型" in results and "等權基準" in results:
        test = tester.sharpe_difference_test(
            results["保守型"].returns,
            results["等權基準"].returns
        )
        
        print(f"\n保守型 vs 等權基準:")
        print(f"  Sharpe差異: {test['difference']:.3f}")
        print(f"  p-value: {test.get('p_value', 'N/A')}")
    
    # 7. 生成圖表
    print("\n步驟6: 生成圖表...")
    try:
        visualizer.plot_cumulative_returns(results, save_path='quick_test_returns.png')
        print("✓ 圖表已儲存: quick_test_returns.png")
    except Exception as e:
        print(f"✗ 圖表生成失敗: {str(e)}")
    
    print("\n" + "="*80)
    print("✓ 快速測試完成！")
    print("="*80)
    
    return results, performance_df

if __name__ == "__main__":
    results, df = quick_test()
