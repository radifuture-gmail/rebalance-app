import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
import json
from datetime import datetime, timedelta, date

# --- 1. ユーティリティ関数（URL保存・復元用） ---

def encode_state(data):
    """JSONデータをBase64エンコードする"""
    if 'start_date' in data and isinstance(data['start_date'], (date, datetime)):
        data['start_date'] = data['start_date'].strftime('%Y-%m-%d')
    json_str = json.dumps(data)
    return base64.b64encode(json_str.encode()).decode()

def decode_state(b64_str):
    """Base64文字列をデコードしてJSONに戻す"""
    try:
        json_str = base64.b64decode(b64_str).decode()
        return json.loads(json_str)
    except Exception as e:
        st.error(f"設定の読み込みに失敗しました: {e}")
        return None

# --- 2. データ取得・計算ロジック ---

@st.cache_data
def get_market_data(tickers, start_date):
    """yfinanceからデータを取得し、USD基準に変換する"""
    if not tickers:
        return pd.DataFrame()

    currencies = ['USDJPY=X', 'EURUSD=X']
    target_tickers = [t for t in tickers if t != 'CASH']
    all_symbols = list(set(target_tickers + currencies))
    
    if not all_symbols:
        return pd.DataFrame()

    try:
        raw_data = yf.download(all_symbols, start=start_date, progress=False, auto_adjust=False, threads=False)
    except Exception as e:
        st.error(f"yfinance download failed: {e}")
        return pd.DataFrame()

    if raw_data.empty:
        return pd.DataFrame()

    # Adj Closeの抽出ロジック
    adj_close = None
    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            adj_close = raw_data['Adj Close']
        except KeyError:
            if 'Close' in raw_data.columns.get_level_values(0):
                 adj_close = raw_data['Close']
    else:
        if 'Adj Close' in raw_data.columns:
            adj_close = raw_data[['Adj Close']]
            if len(all_symbols) == 1: adj_close.columns = [all_symbols[0]]
        elif 'Close' in raw_data.columns:
            adj_close = raw_data[['Close']]
            if len(all_symbols) == 1: adj_close.columns = [all_symbols[0]]

    if adj_close is None or adj_close.empty:
        return pd.DataFrame()

    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()

    # 為替レート
    usdjpy = None
    if 'USDJPY=X' in adj_close.columns:
        usdjpy = adj_close['USDJPY=X'].ffill()

    usd_prices = pd.DataFrame(index=adj_close.index)
    
    for ticker in tickers:
        if ticker == 'CASH':
            usd_prices[ticker] = 1.0
            continue
        if ticker not in adj_close.columns:
            continue
            
        series = adj_close[ticker].ffill()
        
        if ticker.endswith('.T'): 
            if usdjpy is not None:
                usd_prices[ticker] = series / usdjpy
            else:
                usd_prices[ticker] = series
        else:
            usd_prices[ticker] = series

    return usd_prices.dropna()

def calculate_portfolio_performance(prices, weights):
    """リバランスを含めたポートフォリオのトータルリターンを計算"""
    daily_returns = prices.pct_change().fillna(0)
    weighted_returns = pd.DataFrame(index=daily_returns.index)
    
    for ticker, weight in weights.items():
        if ticker in daily_returns.columns:
            weighted_returns[ticker] = daily_returns[ticker] * weight
            
    portfolio_daily_ret = weighted_returns.sum(axis=1)
    cumulative_ret = (1 + portfolio_daily_ret).cumprod()
    return cumulative_ret, portfolio_daily_ret

def calculate_metrics(daily_ret, risk_free_rate=0.0):
    """
    各種指標計算
    - CAGR (年平均成長率)
    - Volatility (年率標準偏差)
    - Sharpe Ratio (シャープレシオ)
    - Max Drawdown (最大下落率)
    - Downside Deviation (下方偏差)
    - Sortino Ratio (ソルティノレシオ)
    """
    ann_factor = 252
    
    # 1. 基本指標
    mean_ret = daily_ret.mean() * ann_factor
    volatility = daily_ret.std() * np.sqrt(ann_factor)
    
    # 2. Sharpe Ratio
    sharpe = 0.0
    if volatility != 0:
        sharpe = (mean_ret - risk_free_rate) / volatility
        
    # 3. Max Drawdown
    cumulative = (1 + daily_ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # 4. Downside Deviation & Sortino Ratio
    # リターンが0未満の日だけをリスクとして計算（Target Return = 0とする）
    negative_rets = daily_ret[daily_ret < 0]
    
    # 下方偏差の計算: (マイナスリターンの二乗和 / 全期間日数) の平方根 * 年率化
    # ※Sortinoの定義により分母は全日数とするのが一般的
    downside_variance = (daily_ret.clip(upper=0) ** 2).mean()
    downside_dev = np.sqrt(downside_variance) * np.sqrt(ann_factor)
    
    sortino = 0.0
    if downside_dev != 0:
        sortino = (mean_ret - risk_free_rate) / downside_dev
        
    return {
        "cagr": mean_ret,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "downside_dev": downside_dev,
        "sortino": sortino
    }

# --- 3. アプリケーション本体 ---

def main():
    st.set_page_config(page_title="Portfolio Backtester", layout="wide")

    # --- URLパラメータ処理 ---
    query_params = st.query_params
    initial_config = None
    if "config" in query_params:
        initial_config = decode_state(query_params["config"])

    # --- セッション初期化 ---
    if 'assets' not in st.session_state:
        default_assets = [
            {'ticker': 'SPY', 'type': 'Long', 'allocation_pct': 50.0},
            {'ticker': 'TLT', 'type': 'Long', 'allocation_pct': 30.0},
        ]
        default_total = 10000.0
        default_start_date = datetime.today() - timedelta(days=365)

        if initial_config:
            st.session_state.total_investment = initial_config.get('total_investment', default_total)
            loaded_assets = initial_config.get('assets', default_assets)
            st.session_state.assets = [a for a in loaded_assets if a.get('type') != 'Cash']
            
            saved_date_str = initial_config.get('start_date')
            if saved_date_str:
                try:
                    st.session_state.start_date = datetime.strptime(saved_date_str, '%Y-%m-%d').date()
                except ValueError:
                    st.session_state.start_date = default_start_date
            else:
                st.session_state.start_date = default_start_date
        else:
            st.session_state.total_investment = default_total
            st.session_state.assets = default_assets
            st.session_state.start_date = default_start_date

    st.title("📈 Global Portfolio Backtester")
    st.markdown("""
    Github連携簡易アプリ。USD基準でトータルリターンを計算します。
    - **Cash自動計算**: 空売りはCash In(現金増)、買いはCash Outとして計算。
    - **詳細分析**: 最大下落率、ソルティノレシオなどを算出。
    """)

    # --- サイドバー ---
    with st.sidebar:
        st.header("Portfolio Settings")
        
        total_inv = st.number_input("Total Investment (USD)", value=float(st.session_state.total_investment), step=1000.0, key='total_input')
        st.session_state.total_investment = total_inv

        start_date_input = st.date_input("Start Date", value=st.session_state.start_date)
        st.session_state.start_date = start_date_input

        st.divider()
        st.subheader("Asset Allocation")

        indices_to_remove = []
        updated_assets = []
        
        for i, asset in enumerate(st.session_state.assets):
            with st.container(border=True):
                col_top1, col_top2 = st.columns([0.85, 0.15])
                with col_top1:
                    st.caption(f"Asset {i+1}")
                with col_top2:
                    if st.button("🗑️", key=f"del_{i}"):
                        indices_to_remove.append(i)

                col1, col2 = st.columns(2)
                with col1:
                    ticker = st.text_input("Ticker", value=asset['ticker'], key=f"tick_{i}", placeholder="e.g. SPY")
                    pos_type = st.selectbox("Type", ["Long", "Short"], index=["Long", "Short"].index(asset.get('type', 'Long')), key=f"type_{i}")
                
                with col2:
                    current_pct = asset.get('allocation_pct', 0.0)
                    new_pct = st.number_input(f"Alloc (%)", value=float(current_pct), step=5.0, key=f"pct_{i}")
                    
                    amount = total_inv * (new_pct/100)
                    if pos_type == 'Short':
                        st.caption(f"Short: +${amount:,.0f}")
                    else:
                        st.caption(f"Long: -${amount:,.0f}")

                updated_assets.append({
                    'ticker': ticker.upper(), 
                    'type': pos_type, 
                    'allocation_pct': new_pct
                })

        if indices_to_remove:
            for index in sorted(indices_to_remove, reverse=True):
                updated_assets.pop(index)
            st.session_state.assets = updated_assets
            st.rerun()

        if st.button("➕ Add Asset"):
            updated_assets.append({'ticker': '', 'type': 'Long', 'allocation_pct': 0.0})
            st.session_state.assets = updated_assets
            st.rerun()

        st.session_state.assets = updated_assets
        st.divider()
        
        if st.button("💾 Save Config to URL"):
            config_to_save = {
                'total_investment': st.session_state.total_investment,
                'assets': st.session_state.assets,
                'start_date': st.session_state.start_date
            }
            encoded = encode_state(config_to_save)
            st.query_params["config"] = encoded
            st.success("Config saved to URL!")

    # --- メインエリア ---

    st.subheader("1. Portfolio Composition & Cash Calculation")
    
    if not st.session_state.assets:
        st.warning("Please add assets in the sidebar.")
        return

    df_assets = pd.DataFrame(st.session_state.assets)
    df_display = df_assets[df_assets['ticker'] != ''].copy()

    if df_display.empty:
        st.info("Enter tickers to begin.")
        return

    # Cash計算
    long_pct = df_display[df_display['type'] == 'Long']['allocation_pct'].sum()
    short_pct = df_display[df_display['type'] == 'Short']['allocation_pct'].sum()
    calculated_cash_pct = 100.0 - long_pct + short_pct
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    with col_metrics1:
        st.metric("Total Long", f"{long_pct:.1f}%")
    with col_metrics2:
        st.metric("Total Short", f"{short_pct:.1f}%")
    with col_metrics3:
        cash_val = total_inv * (calculated_cash_pct/100)
        st.metric("Implied Cash", f"{calculated_cash_pct:.1f}%", f"${cash_val:,.0f}")

    if calculated_cash_pct < 0:
        st.error(f"⚠️ **Leverage Warning**: Cash is negative ({calculated_cash_pct:.1f}%).")

    # 円グラフ
    pie_data = df_display[['ticker', 'allocation_pct', 'type']].copy()
    if calculated_cash_pct != 0:
        pie_data = pd.concat([pie_data, pd.DataFrame([{
            'ticker': 'CASH (USD)', 'allocation_pct': abs(calculated_cash_pct), 
            'type': 'Cash' if calculated_cash_pct > 0 else 'Debt'
        }])], ignore_index=True)

    col_pie1, col_pie2 = st.columns(2)
    with col_pie1:
        fig_pie = px.pie(pie_data, values='allocation_pct', names='ticker', hole=0.4, title="Allocation (Abs %)")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_pie2:
        df_type = pie_data.groupby('type')['allocation_pct'].sum().reset_index()
        fig_pie_type = px.pie(df_type, values='allocation_pct', names='type', color='type', title="Asset Exposure",
                             color_discrete_map={'Long':'#00CC96', 'Short':'#EF553B', 'Cash':'#636EFA'})
        st.plotly_chart(fig_pie_type, use_container_width=True)

    # 2. バックテスト
    st.subheader("2. Historical Performance")
    
    if st.button("🚀 Run Backtest"):
        with st.spinner("Calculating..."):
            tickers_to_fetch = [t for t in df_display['ticker'].unique() if t.strip() != '']
            
            try:
                prices_df = get_market_data(tickers_to_fetch, st.session_state.start_date)
            except Exception as e:
                st.error(str(e))
                return

            if prices_df.empty:
                st.error("No data found.")
                return

            # ウェイト計算
            weights = {}
            for _, row in df_display.iterrows():
                w = row['allocation_pct'] / 100.0
                if row['type'] == 'Short':
                    w = -1.0 * w 
                weights[row['ticker']] = w
            
            current_asset_weight_sum = sum(weights.values())
            weights['CASH'] = 1.0 - current_asset_weight_sum

            calc_df = prices_df.copy()
            if 'CASH' not in calc_df.columns:
                if calc_df.empty:
                     daterange = pd.date_range(start=st.session_state.start_date, end=datetime.today())
                     calc_df = pd.DataFrame(index=daterange)
                calc_df['CASH'] = 1.0
            else:
                calc_df['CASH'] = 1.0

            cumulative_returns, daily_returns = calculate_portfolio_performance(calc_df, weights)
            equity_curve = cumulative_returns * total_inv
            
            # --- 指標計算の実行 ---
            metrics = calculate_metrics(daily_returns)

            # 結果表示 (KPIカード)
            st.markdown("### 📊 Performance Metrics")
            # 6つの指標を2行または1行で表示
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            final_val = equity_curve.iloc[-1]
            total_ret_pct = (final_val / total_inv - 1) * 100
            
            m1.metric("Final Value", f"${final_val:,.0f}", f"{total_ret_pct:.1f}%")
            m2.metric("CAGR", f"{metrics['cagr']*100:.2f}%", help="Annualized Return")
            m3.metric("Volatility", f"{metrics['volatility']*100:.2f}%", help="Annualized Standard Deviation")
            m4.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}", help="Return / Volatility")
            m5.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%", help="Maximum peak-to-trough decline")
            m6.metric("Sortino Ratio", f"{metrics['sortino']:.2f}", help="Return / Downside Deviation")

            # チャート
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Total Portfolio', line=dict(width=2, color='#636EFA')))
            fig.update_layout(title='Portfolio Value (USD)', xaxis_title='Date', yaxis_title='Value ($)', hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("3. Individual Asset Performance (Base=100)")
            if not prices_df.empty:
                disp_cols = [c for c in prices_df.columns if c != 'CASH']
                if disp_cols:
                    normalized_df = prices_df[disp_cols] / prices_df[disp_cols].iloc[0] * 100
                    fig_ind = px.line(normalized_df, title="Asset Performance")
                    st.plotly_chart(fig_ind, use_container_width=True)

if __name__ == "__main__":
    main()
