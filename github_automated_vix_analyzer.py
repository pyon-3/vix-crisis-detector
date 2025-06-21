#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GitHub Actions Automated VIX Analysis System
日経VIなしでVIX・日経・S&P500のみで高度分析
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import warnings
from arch import arch_model
import os

warnings.filterwarnings('ignore')

class GitHubVIXAnalyzer:
    def __init__(self):
        self.data = None
        self.garch_model = None
        self.risk_score = 0
        self.risk_level = ""
        
    def download_data(self, period='1y'):
        """データ取得（日経VIなし）"""
        print("📊 市場データ取得中...")
        
        # VIX, 日経, S&P500のみ
        vix = yf.download('^VIX', period=period, auto_adjust=True, threads=True)
        nikkei = yf.download('^N225', period=period, auto_adjust=True, threads=True)
        sp500 = yf.download('^GSPC', period=period, auto_adjust=True, threads=True)
        
        if vix.empty or nikkei.empty or sp500.empty:
            raise ValueError("データ取得に失敗しました")
        
        # データフレーム構築
        data = pd.DataFrame(index=vix.index)
        data['VIX'] = vix['Close']
        data['Nikkei'] = nikkei['Close']
        data['SP500'] = sp500['Close']
        
        # リターン計算
        data['Nikkei_Returns'] = data['Nikkei'].pct_change()
        data['SP500_Returns'] = data['SP500'].pct_change()
        data['VIX_Returns'] = data['VIX'].pct_change()
        
        # 実現ボラティリティ計算（日経VI代替）
        data['Nikkei_RV_5d'] = data['Nikkei_Returns'].rolling(5).std() * np.sqrt(252) * 100
        data['Nikkei_RV_20d'] = data['Nikkei_Returns'].rolling(20).std() * np.sqrt(252) * 100
        
        # VIX vs 実現ボラ スプレッド
        data['VIX_RV_Spread'] = data['VIX'] - data['Nikkei_RV_20d']
        
        # 相対的強弱
        data['Nikkei_SP500_Ratio'] = data['Nikkei'] / data['SP500']
        data['Ratio_MA20'] = data['Nikkei_SP500_Ratio'].rolling(20).mean()
        data['Ratio_Deviation'] = data['Nikkei_SP500_Ratio'] - data['Ratio_MA20']
        
        self.data = data.dropna()
        print(f"✅ データ取得完了: {len(self.data)}日分")
        return self.data
    
    def fit_garch_model(self):
        """GARCH(1,1)モデル推定"""
        print("🔬 GARCH(1,1)モデル推定中...")
        
        returns = self.data['Nikkei_Returns'].dropna() * 100
        
        try:
            self.garch_model = arch_model(returns, vol='Garch', p=1, q=1, 
                                         mean='Constant', dist='normal')
            garch_result = self.garch_model.fit(disp='off')
            
            conditional_vol = garch_result.conditional_volatility
            self.data['GARCH_Vol'] = np.nan
            self.data.loc[conditional_vol.index, 'GARCH_Vol'] = conditional_vol
            self.data['GARCH_Vol_Annualized'] = self.data['GARCH_Vol'] * np.sqrt(252)
            
            print(f"✅ GARCH推定完了 - 現在予測ボラ: {self.data['GARCH_Vol_Annualized'].iloc[-1]:.2f}%")
            return garch_result
        except Exception as e:
            print(f"⚠️ GARCH推定失敗: {e}")
            return None
    
    def calculate_risk_indicators(self):
        """リスク指標計算"""
        print("📊 リスク指標計算中...")
        
        # VIX指標
        self.data['VIX_MA20'] = self.data['VIX'].rolling(20).mean()
        self.data['VIX_Spike'] = self.data['VIX'] > (self.data['VIX_MA20'] + self.data['VIX'].rolling(20).std())
        
        # ボラティリティ・レジーム
        if 'GARCH_Vol_Annualized' in self.data.columns:
            garch_quantiles = self.data['GARCH_Vol_Annualized'].quantile([0.33, 0.67])
            self.data['Vol_Regime'] = np.where(
                self.data['GARCH_Vol_Annualized'] > garch_quantiles[0.67], 'High',
                np.where(self.data['GARCH_Vol_Annualized'] < garch_quantiles[0.33], 'Low', 'Medium')
            )
        
        # 相対強弱過熱
        self.data['Ratio_Extreme'] = abs(self.data['Ratio_Deviation']) > self.data['Ratio_Deviation'].std() * 2
        
        print("✅ リスク指標計算完了")
    
    def generate_signals(self):
        """シグナル生成"""
        print("🚨 シグナル生成中...")
        
        # 警告条件
        vix_elevated = self.data['VIX'] > 25
        vix_rv_spread_high = self.data['VIX_RV_Spread'] > self.data['VIX_RV_Spread'].quantile(0.8)
        ratio_extreme = self.data['Ratio_Extreme']
        
        if 'Vol_Regime' in self.data.columns:
            vol_regime_high = self.data['Vol_Regime'] == 'High'
        else:
            vol_regime_high = pd.Series(False, index=self.data.index)
        
        # 複合シグナル
        signal_count = (vix_elevated.astype(int) + 
                       vix_rv_spread_high.astype(int) + 
                       ratio_extreme.astype(int) + 
                       vol_regime_high.astype(int))
        
        self.data['Warning_Signal'] = signal_count >= 2
        self.data['Crash_Signal'] = signal_count >= 3
        
        print("✅ シグナル生成完了")
    
    def calculate_risk_score(self):
        """総合リスクスコア計算"""
        latest = self.data.iloc[-1]
        
        # 各指標のスコア計算 (0-100)
        vix_score = min(latest['VIX'] / 40 * 100, 100)
        
        if 'GARCH_Vol_Annualized' in self.data.columns:
            garch_score = min(latest['GARCH_Vol_Annualized'] / 30 * 100, 100)
        else:
            garch_score = min(latest['Nikkei_RV_20d'] / 30 * 100, 100)
        
        spread_score = min(abs(latest['VIX_RV_Spread']) / 15 * 100, 100)
        ratio_score = min(abs(latest['Ratio_Deviation']) * 1000, 100)
        
        # 総合スコア
        self.risk_score = (vix_score + garch_score + spread_score + ratio_score) / 4
        
        # レベル判定
        if self.risk_score > 70:
            self.risk_level = "🔴 HIGH RISK"
        elif self.risk_score > 40:
            self.risk_level = "🟡 MEDIUM RISK"
        else:
            self.risk_level = "🟢 LOW RISK"
        
        print(f"🎯 リスクスコア: {self.risk_score:.1f}/100 - {self.risk_level}")
        
        return {
            'total_score': self.risk_score,
            'level': self.risk_level,
            'vix_score': vix_score,
            'volatility_score': garch_score,
            'spread_score': spread_score,
            'ratio_score': ratio_score
        }
    
    def create_dashboard(self, save_path="docs"):
        """GitHub Pages用ダッシュボード作成"""
        print("📈 GitHub Pages用ダッシュボード作成中...")
        
        # ディレクトリ作成
        os.makedirs(save_path, exist_ok=True)
        
        # Plotly ダッシュボード
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'VIX Fear Index Overview',
                'Risk Score Dashboard',
                'VIX vs Realized Volatility',
                'Nikkei vs SP500 Relative Strength',
                'Signal History',
                'Current Risk Assessment'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. VIX Overview
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['VIX'],
                      name='VIX恐怖指数', line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # VIX レベル参考線を追加
        fig.add_hline(y=20, line_dash="dash", line_color="orange", 
                     annotation_text="警戒水準(20)", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="red", 
                     annotation_text="高警戒水準(30)", row=1, col=1)
        
        # 2. Risk Score (Bar Chart代替)
        latest = self.data.iloc[-1]
        risk_data = self.calculate_risk_score()
        
        risk_components = ['VIX Score', 'Vol Score', 'Spread Score', 'Ratio Score']
        risk_values = [risk_data['vix_score'], risk_data['volatility_score'], 
                      risk_data['spread_score'], risk_data['ratio_score']]
        
        fig.add_trace(
            go.Bar(x=risk_components, y=risk_values,
                   name='リスクスコア構成',
                   marker_color=['red' if v > 70 else 'orange' if v > 40 else 'green' for v in risk_values]),
            row=1, col=2
        )
        
        # Total Risk Score を注釈で表示
        fig.add_annotation(
            text=f"<b>Total Risk: {self.risk_score:.1f}/100</b><br>{self.risk_level}",
            xref="x domain", yref="y domain",
            x=0.5, y=0.9,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=16, color="red" if self.risk_score > 70 else "orange" if self.risk_score > 40 else "green"),
            bgcolor="rgba(255,255,255,0.8)",
            row=1, col=2
        )
        
        # 3. VIX vs RV
        if 'GARCH_Vol_Annualized' in self.data.columns:
            fig.add_trace(
                go.Scatter(x=self.data.index, y=self.data['GARCH_Vol_Annualized'],
                          name='GARCH予測ボラ', line=dict(color='blue')),
                row=2, col=1
            )
        
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['Nikkei_RV_20d'],
                      name='実現ボラ20日', line=dict(color='green')),
            row=2, col=1
        )
        
        # 4. Relative Strength
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['Nikkei_SP500_Ratio'],
                      name='日経/SP500比率', line=dict(color='purple')),
            row=2, col=2
        )
        
        # 5. Signals
        warning_dates = self.data[self.data['Warning_Signal']].index
        crash_dates = self.data[self.data['Crash_Signal']].index
        
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['VIX'],
                      name='VIX', line=dict(color='black')),
            row=3, col=1
        )
        
        if len(warning_dates) > 0:
            fig.add_trace(
                go.Scatter(x=warning_dates, y=self.data.loc[warning_dates, 'VIX'],
                          mode='markers', name='警告シグナル',
                          marker=dict(color='orange', size=8)),
                row=3, col=1
            )
        
        if len(crash_dates) > 0:
            fig.add_trace(
                go.Scatter(x=crash_dates, y=self.data.loc[crash_dates, 'VIX'],
                          mode='markers', name='クラッシュシグナル',
                          marker=dict(color='red', size=10, symbol='triangle-down')),
                row=3, col=1
            )
        
        # 6. Current Assessment
        current_metrics = {
            'VIX': latest['VIX'],
            'Nikkei RV': latest['Nikkei_RV_20d'],
            'VIX/RV Spread': latest['VIX_RV_Spread']
        }
        
        fig.add_trace(
            go.Bar(x=list(current_metrics.keys()), y=list(current_metrics.values()),
                   name='現在の指標'),
            row=3, col=2
        )
        
        # レイアウト
        fig.update_layout(
            height=1000,
            title_text="<b>🔬 Automated VIX Risk Analysis Dashboard</b>",
            title_font_size=24,
            showlegend=True
        )
        
        # ウォーターマーク
        fig.add_annotation(
            text="@pyon - Automated Analysis",
            xref="paper", yref="paper",
            x=0.02, y=0.02,
            showarrow=False,
            font=dict(size=24, color="rgba(128,128,128,0.7)"),
            align="left"
        )
        
        # HTML保存
        html_file = os.path.join(save_path, "index.html")
        fig.write_html(html_file)
        print(f"✅ ダッシュボード保存: {html_file}")
        
        return html_file
    
    def generate_json_report(self, save_path="docs"):
        """JSON レポート生成"""
        latest = self.data.iloc[-1]
        latest_date = self.data.index[-1].strftime('%Y-%m-%d')
        
        risk_data = self.calculate_risk_score()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_date": latest_date,
            "risk_assessment": {
                "total_score": round(self.risk_score, 1),
                "level": self.risk_level,
                "components": {
                    "vix_score": round(risk_data['vix_score'], 1),
                    "volatility_score": round(risk_data['volatility_score'], 1),
                    "spread_score": round(risk_data['spread_score'], 1),
                    "ratio_score": round(risk_data['ratio_score'], 1)
                }
            },
            "current_metrics": {
                "vix": round(latest['VIX'], 2),
                "nikkei": round(latest['Nikkei'], 2),
                "sp500": round(latest['SP500'], 2),
                "nikkei_rv_20d": round(latest['Nikkei_RV_20d'], 2),
                "vix_rv_spread": round(latest['VIX_RV_Spread'], 2),
                "nikkei_sp500_ratio": round(latest['Nikkei_SP500_Ratio'], 4)
            },
            "signals": {
                "warning_signal": int(latest['Warning_Signal']),
                "crash_signal": int(latest['Crash_Signal'])
            },
            "alert_required": bool(self.risk_score > 60 or latest['Crash_Signal'])
        }
        
        # JSON保存
        os.makedirs(save_path, exist_ok=True)
        json_file = os.path.join(save_path, "risk_report.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ JSONレポート保存: {json_file}")
        return report

def main():
    """メイン実行関数"""
    analyzer = GitHubVIXAnalyzer()
    
    # データ取得・分析
    analyzer.download_data(period='1y')
    analyzer.fit_garch_model()
    analyzer.calculate_risk_indicators()
    analyzer.generate_signals()
    
    # ダッシュボード・レポート作成
    analyzer.create_dashboard()
    report = analyzer.generate_json_report()
    
    # 結果表示
    print(f"\n{'='*50}")
    print(f"🔬 AUTOMATED VIX ANALYSIS REPORT")
    print(f"{'='*50}")
    print(f"📅 分析日: {report['analysis_date']}")
    print(f"🎯 リスクスコア: {report['risk_assessment']['total_score']}/100")
    print(f"📊 リスクレベル: {report['risk_assessment']['level']}")
    print(f"🚨 アラート必要: {'YES' if report['alert_required'] else 'NO'}")
    print(f"{'='*50}")
    
    return analyzer, report

if __name__ == "__main__":
    analyzer, report = main()