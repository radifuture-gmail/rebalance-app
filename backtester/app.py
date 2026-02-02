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
    # 日付オブジェクトが含まれる場合、文字列に変換
    if 'start_date' in data and isinstance(data['start_date'], (date, datetime)):
        data['start_date'] = data['start_date'].strftime('%Y-%m-%d')
        
    json_str = json.dumps(data)
    return base64.b64encode(json_str.encode()).decode()

def decode_state(b64_str):
    """Base64文字列をデコードしてJSONに戻す"""
    try:
        json_str = base64.b64decode(b64_str).decode()
        data = json.loads(json_str)
        return data
    except Exception as e:
        st.error(f"設定の読み込みに失敗しました: {e}")
        return None

# --- 2. データ取得・計算ロジック ---

@st.cache_data
def get_market_data(tickers, start_date):
    """
    yfinanceからデータを取得し、USD基準に変換する。
    堅牢性を高めるため、データ取得とカラム抽出を分離。
    """
    if not tickers:
        return pd.DataFrame()

    # 為替レートの取得 (USDJPYなど必要に応じて追加)
    currencies = ['USDJPY=X', 'EURUSD=X'] 
    
    # ユーザー入力のTickerと為替を合わせて取得
    # Cashは除外済みのはずだが念のためフィルタ
    target_tickers = [t for t in tickers if t != 'CASH']
    all_symbols = list(set(target_tickers + currencies))
    
    if not all_symbols:
        return pd.DataFrame()

    try:
        # 1. データを一括ダウンロード (auto_adjust=Falseで'Adj Close'を確実に取得)
        # threads=False は一部環境でのエラーを防ぐために設定
        raw_data = yf.download(all_symbols, start=start_date, progress=False, auto_adjust=False, threads=False)
    except Exception as e:
        st.error(f"yfinance download failed: {e}")
        return pd.DataFrame()

    # 2. データが空でないか確認
    if raw_data.empty:
        return pd.DataFrame()

    # 3. 'Adj Close' カラムを安全に取り出す
    adj_close = None
    
    # ケースA: MultiIndexの場合 (Columns = [('Adj Close', 'SPY'), ...])
    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            adj_close = raw_data['Adj Close']
        except KeyError:
            # Adj Closeがない場合はCloseで代用を試みる
            if 'Close' in raw_data.columns.get_level_values(0):
                 adj_close = raw_data['Close']
    
    # ケースB: 単一Indexの場合 (Columns = ['Adj Close', 'Open', ...])
    else:
        if 'Adj Close' in raw_data.columns:
            adj_close = raw_data[['Adj Close']]
            # カラム名が 'Adj Close' のままかもしれないので、銘柄名に変更を試みる
            if len(all_symbols) == 1:
                adj_close.columns = [all_symbols[0]]
        elif 'Close' in raw_data.columns:
            adj_close = raw_data[['Close']]
            if len(all_symbols) == 1:
                adj_close.columns = [all_symbols[0]]

    if adj_close is None or adj_close.empty:
        return pd.DataFrame()

    # Seriesの場合はDataFrameに変換
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()

    # 4. 為替レートの準備
    usdjpy = None
    if 'USDJPY=X' in adj_close.columns:
        usdjpy = adj_close['USDJPY=X'].ffill()

    # 5. USD基準の価格データフレームを作成
    usd_prices = pd.DataFrame(index=adj_close.index)
    
    for ticker in tickers:
        if ticker == 'CASH':
            usd_prices[ticker] = 1.0
            continue
            
        if ticker not in adj_close.columns:
            continue
            
        series = adj_close[ticker].ffill()
        
        # 簡易的な通貨判定ロジック (.T = Japan)
        if ticker.endswith('.T'): 
            if usdjpy is not None:
                usd_prices[ticker] = series / usdjpy
            else:
                usd_prices[ticker] = series
        else:
            usd_prices[ticker] = series

    return usd_prices.dropna()

def calculate_portfolio_performance(prices, weights):
    """
    リバランスを含めたポートフォリオのトータルリターンを計算
    """
    # 価格データから日次リターンを計算
    daily_returns = prices.pct_change().fillna(0)
    
    weighted_returns = pd.DataFrame(index=daily_returns.index)
    
    for ticker, weight in weights.items():
        if ticker in daily_returns.columns:
            # ショートの場合はウェイトが負になっている想定
            weighted_returns[ticker] = daily_returns[ticker] * weight
            
    # ポートフォリオ全体の日次リターン
    portfolio_daily_ret = weighted_returns.sum(axis=1)
    
    # 累積リターン (1 + r).cumprod()
    cumulative_ret = (1 + portfolio_daily_ret).cumprod()
    
    return cumulative_ret

# --- 3. アプリケーション本体 ---

def main():
    st.set_page_config(page_title="Portfolio Backtester", layout="wide")

    # --- URLパラメータの読み込み ---
    query_params = st.query_params
    initial_config = None
    
    if "config" in query_params:
        initial_config = decode_state(query_params["config"])

    # --- セッション状態の初期化 ---
    if 'assets' not in st.session_state:
        # デフォルト値
        default_assets = [
            {'ticker': 'SPY', 'type': 'Long', 'allocation_pct': 50.0},
            {'ticker': 'TLT', 'type': 'Long', 'allocation_pct': 30.0},
            {'ticker': 'CASH', 'type': 'Cash', 'allocation_pct': 20.0},
        ]
        default_total = 10000.0
        default_start_date = datetime.today() - timedelta(days=365)

        if initial_config:
            st.session_state.total_investment = initial_config.get('total_investment', default_total)
            st.session_state.assets = initial_config.get('assets', default_assets)
            
            # 日付の復元処理
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
    - **日本株対応**: `.T` をつけるとUSD換算されます。
    - **トータルリターン**: 配当込み（Adj Close）を使用。
    - **URL保存**: 構成比率だけでなく、**Start Date**も保存されます。
    """)

    # --- サイドバー：設定 ---
    with st.sidebar:
        st.header("Portfolio Settings")
        
        # 総資金額
        total_inv = st.number_input(
            "Total Investment (USD)", 
            value=float(st.session_state.total_investment), 
            step=1000.0,
            key='total_input'
        )
        st.session_state.total_investment = total_inv

        # Start Dateの入力（session_stateと連動）
        start_date_input = st.date_input(
            "Start Date", 
            value=st.session_state.start_date
        )
        st.session_state.start_date = start_date_input

        st.divider()
        st.subheader("Asset Allocation")

        updated_assets = []
        
        # 既存のアセットを表示・編集
        for i, asset in enumerate(st.session_state.assets):
            with st.expander(f"Asset {i+1}: {asset.get('ticker', '')}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    ticker = st.text_input("Ticker (e.g., AAPL, 7203.T)", value=asset['ticker'], key=f"tick_{i}")
                    pos_type = st.selectbox("Type", ["Long", "Short", "Cash"], index=["Long", "Short", "Cash"].index(asset.get('type', 'Long')), key=f"type_{i}")
                
                with col2:
                    current_pct = asset.get('allocation_pct', 0.0)
                    new_pct = st.number_input(f"Allocation (%)", value=float(current_pct), step=1.0, key=f"pct_{i}")
                    st.caption(f"Amount: ${total_inv * (new_pct/100):,.2f}")
            
            # 空文字でも入力途中として保持する（以前の修正）
            updated_assets.append({
                'ticker': ticker.upper(), 
                'type': pos_type, 
                'allocation_pct': new_pct
            })

        if st.button("➕ Add Asset"):
            updated_assets.append({'ticker': '', 'type': 'Long', 'allocation_pct': 0.0})
            st.session_state.assets = updated_assets
            st.rerun()
            
        if st.button("🗑️ Remove Last"):
            if updated_assets:
                updated_assets.pop()
                st.session_state.assets = updated_assets
                st.rerun()

        st.session_state.assets = updated_assets

        st.divider()
        
        # 保存ボタン（Start Dateも含める）
        if st.button("💾 Save Config to URL"):
            config_to_save = {
                'total_investment': st.session_state.total_investment,
                'assets': st.session_state.assets,
                'start_date': st.session_state.start_date # Dateオブジェクト
            }
            encoded = encode_state(config_to_save)
            st.query_params["config"] = encoded
            st.success("Configuration (including Start Date) saved to URL!")

    # --- メインエリア ---

    # 1. 構成確認
    st.subheader("1. Portfolio Composition")
    
    if not st.session_state.assets:
        st.warning("Please add assets in the sidebar.")
        return

    df_assets = pd.DataFrame(st.session_state.assets)
    # Ticker未入力がある場合の処理
    df_display = df_assets[df_assets['ticker'] != ''].copy()
    
    if df_display.empty:
        st.info("Enter tickers to begin.")
        return

    df_display['amount'] = (df_display['allocation_pct'] / 100) * total_inv
    st.dataframe(df_display.style.format({'allocation_pct': '{:.2f}%', 'amount': '${:,.2f}'}))

    total_pct = df_display['allocation_pct'].sum()
    if not (99.9 <= total_pct <= 100.1):
        st.warning(f"⚠️ Total allocation is {total_pct:.2f}%. It is recommended to sum to 100%.")

    col_pie1, col_pie2 = st.columns(2)
    with col_pie1:
        fig_pie = px.pie(df_display, values='allocation_pct', names='ticker', hole=0.4, title="Allocation by Ticker")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_pie2:
        df_type = df_display.groupby('type')['allocation_pct'].sum().reset_index()
        fig_pie_type = px.pie(df_type, values='allocation_pct', names='type', color='type', title="Long / Short / Cash",
                             color_discrete_map={'Long':'#00CC96', 'Short':'#EF553B', 'Cash':'#636EFA'})
        st.plotly_chart(fig_pie_type, use_container_width=True)

    # 2. バックテスト
    st.subheader("2. Historical Performance")
    
    if st.button("🚀 Run Backtest"):
        with st.spinner("Fetching data and calculating..."):
            
            # TypeがCashのものはyfinanceに投げない
            tickers_to_fetch = [
                a['ticker'] for a in st.session_state.assets 
                if a['ticker'].strip() != '' and a['type'] != 'Cash'
            ]
            
            # データ取得
            try:
                if not tickers_to_fetch:
                     st.warning("No investable assets (Long/Short) found to fetch data for.")
                     prices_df = pd.DataFrame() # 空のDF
                else:
                    # session_stateの日付を使用
                    prices_df = get_market_data(tickers_to_fetch, st.session_state.start_date)
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return

            if prices_df.empty and tickers_to_fetch:
                st.error("No data found for the specified tickers.")
                return

            # --- A. ポートフォリオ全体の計算 ---
            weights = {}
            for asset in st.session_state.assets:
                if asset['ticker'] == '': continue
                w = asset['allocation_pct'] / 100.0
                if asset['type'] == 'Short':
                    w = -1 * w
                weights[asset['ticker']] = w

            # prices_dfにはCashが含まれていない可能性があるため、Cashがある場合は計算用に補完
            calc_df = prices_df.copy()
            if 'CASH' not in calc_df.columns:
                 # データフレームの長さに合わせた1.0の列を作成
                 # calc_dfが空（CashのみのPF）の場合の対処
                 if calc_df.empty: 
                     # 日付範囲を作成
                     daterange = pd.date_range(start=st.session_state.start_date, end=datetime.today())
                     calc_df = pd.DataFrame(index=daterange)
                 
                 calc_df['CASH'] = 1.0

            cumulative_returns = calculate_portfolio_performance(calc_df, weights)
            equity_curve = cumulative_returns * total_inv
            
            st.metric("Final Portfolio Value", f"${equity_curve.iloc[-1]:,.2f}", 
                      delta=f"{(equity_curve.iloc[-1]/total_inv - 1)*100:.2f}%")

            # チャート1: ポートフォリオ全体
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, mode='lines', name='Total Portfolio', line=dict(width=3, color='royalblue')))
            fig.update_layout(title='Portfolio Value (USD)', xaxis_title='Date', yaxis_title='Value ($)', hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # --- B. 個別銘柄のパフォーマンス（正規化） ---
            st.subheader("3. Individual Asset Performance (Normalized)")
            st.caption("Each asset is normalized to 100 at the start date to compare relative performance in USD.")

            if not prices_df.empty:
                # 100スタートに正規化 (価格 / 初日の価格 * 100)
                # 注: Shortポジションの銘柄も、ここでは「資産そのものの値動き」を表示するのが一般的です（下落すればチャートも下がる）
                normalized_df = prices_df / prices_df.iloc[0] * 100
                
                fig_ind = px.line(normalized_df, title="Asset Performance (Base=100)")
                fig_ind.update_layout(xaxis_title='Date', yaxis_title='Normalized Return', hovermode="x unified")
                st.plotly_chart(fig_ind, use_container_width=True)
            else:
                st.info("No market data available to show individual performance (Only Cash?).")

if __name__ == "__main__":
    main()