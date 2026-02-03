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

# --- 2. データ取得ロジック ---

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

    # Adj Close抽出
    adj_close = None
    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            adj_close = raw_data['Adj Close']
        except KeyError:
            if 'Close' in raw_data.columns.get_level_values(0):
                 adj_close = raw_data['Close']
    else:
        # Single Index
        col_names = list(raw_data.columns)
        if 'Adj Close' in col_names:
            adj_close = raw_data[['Adj Close']]
            if len(all_symbols) == 1: adj_close.columns = [all_symbols[0]]
        elif 'Close' in col_names:
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

# --- 3. バックテスト計算ロジック（リバランス対応版） ---

def run_backtest(prices, weights_config, initial_capital, rebalance_freq, margin_ratios):
    """
    指定されたリバランス頻度でシミュレーションを行う
    prices: 日次価格データ
    weights_config: {ticker: target_weight (+/-)}
    rebalance_freq: 'None', 'D', 'W', 'M', 'Q', 'BA', 'A'
    margin_ratios: {ticker: ratio (0.0~1.0+)}
    """
    if prices.empty:
        return None, None, None

    # 日付インデックスを正規化
    prices.index = pd.to_datetime(prices.index)
    
    # 結果格納用
    dates = prices.index
    portfolio_values = np.zeros(len(dates))
    required_margins = np.zeros(len(dates))
    rebalance_flags = np.zeros(len(dates), dtype=bool) # リバランス実施日
    
    # 現在の保有口数 (shares)
    current_shares = {ticker: 0.0 for ticker in weights_config.keys()}
    cash_holdings = initial_capital # Cashウェイト分もここに含む実質現金

    # リバランス日の判定用
    if rebalance_freq == 'None':
        # 初日のみリバランス
        rb_dates = [dates[0]]
    else:
        # 頻度文字列の変換 (Pandas offset alias)
        freq_map = {
            'Daily': 'D', 'Weekly': 'W-FRI', 'Monthly': 'ME', 
            'Quarterly': 'QE', 'Semi-Annually': '6ME', 'Annually': 'YE'
        }
        freq_str = freq_map.get(rebalance_freq, 'D')
        
        # リバランス予定日を生成
        rb_dates_idx = pd.date_range(start=dates[0], end=dates[-1], freq=freq_str)
        
        # データに存在する直近の営業日にマッピング
        # asofを使って、各予定日以前の最新営業日を探す、あるいは単に日付のマッチングを行う
        # ここでは簡易的に「予定日以降の最初の営業日」をリバランス日とする
        rb_dates = []
        # 初日は必ず実行
        rb_dates.append(dates[0])
        
        # 2回目以降
        search_idx = 0
        for target_date in rb_dates_idx:
            if target_date <= dates[0]: continue
            # target_date以降のデータを探す
            future_dates = dates[dates >= target_date]
            if not future_dates.empty:
                next_date = future_dates[0]
                if next_date not in rb_dates:
                    rb_dates.append(next_date)
    
    rb_dates_set = set(rb_dates)

    # --- 日次ループ ---
    # ベクトル化も可能だが、リバランスロジックの可読性のためループ処理
    
    for i, date in enumerate(dates):
        price_row = prices.iloc[i]
        
        # 1. リバランス判定
        if date in rb_dates_set:
            rebalance_flags[i] = True
            
            # リバランス直前の総資産額を計算
            # (Shares * Price) + Cash
            current_equity = cash_holdings
            for ticker, shares in current_shares.items():
                if ticker == 'CASH': continue # Cashはshares管理しない
                current_equity += shares * price_row[ticker]
            
            # 目標構成比に基づいて再配分
            # Cashポジションは計算上残余として扱うが、weights_configには'CASH'キーが含まれている想定
            
            new_shares = {}
            new_cash = current_equity # 一旦全額現金化の概念
            
            # 株式等の購入/空売り
            for ticker, target_w in weights_config.items():
                if ticker == 'CASH':
                    continue # Cashは最後に残る
                
                target_amt = current_equity * target_w # Shortならマイナス金額
                price = price_row[ticker]
                
                if price != 0:
                    shares = target_amt / price
                    new_shares[ticker] = shares
                    new_cash -= target_amt # 買った分減る、売った(空売り)分増える
            
            current_shares = new_shares
            cash_holdings = new_cash
        
        # 2. その日の資産評価額計算
        daily_equity = cash_holdings
        daily_margin_req = 0.0
        
        for ticker, shares in current_shares.items():
            if ticker == 'CASH': continue
            
            val = shares * price_row[ticker]
            daily_equity += val
            
            # 必要証拠金計算 (|Position Value| * Margin Ratio)
            m_ratio = margin_ratios.get(ticker, 0.0)
            daily_margin_req += abs(val) * (m_ratio / 100.0)
            
        portfolio_values[i] = daily_equity
        required_margins[i] = daily_margin_req

    # Seriesに変換
    portfolio_series = pd.Series(portfolio_values, index=dates)
    margin_series = pd.Series(required_margins, index=dates)
    
    # 日次リターン計算
    daily_returns = portfolio_series.pct_change().fillna(0)
    
    return portfolio_series, margin_series, daily_returns, rebalance_flags

def calculate_metrics(daily_ret, risk_free_rate_pct=0.0):
    """各種指標計算"""
    ann_factor = 252
    rf = risk_free_rate_pct / 100.0
    
    # CAGR
    total_ret = (1 + daily_ret).prod() - 1
    # 期間が短い場合の補正が必要だが、簡易的に
    n_years = len(daily_ret) / ann_factor
    if n_years > 0:
        cagr = (1 + total_ret) ** (1/n_years) - 1
    else:
        cagr = 0.0

    # Volatility
    volatility = daily_ret.std() * np.sqrt(ann_factor)
    
    # Sharpe Ratio
    # (Rp - Rf) / Sigma. Rfは日次に変換して引くのが一般的だが、簡易的に年率で計算
    sharpe = 0.0
    if volatility != 0:
        sharpe = (cagr - rf) / volatility
        
    # Max Drawdown
    cumulative = (1 + daily_ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # Sortino Ratio
    # Rfを考慮した下方偏差
    daily_rf = (1 + rf)**(1/ann_factor) - 1
    excess_ret = daily_ret - daily_rf
    downside_ret = excess_ret[excess_ret < 0]
    
    downside_std = np.sqrt((downside_ret**2).mean()) * np.sqrt(ann_factor)
    
    sortino = 0.0
    if downside_std != 0:
        sortino = (cagr - rf) / downside_std
        
    return {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "sortino": sortino
    }

# --- 4. アプリケーション本体 ---

def main():
    st.set_page_config(page_title="Portfolio Backtester Pro", layout="wide")

    # --- URLパラメータ処理 ---
    query_params = st.query_params
    initial_config = None
    if "config" in query_params:
        initial_config = decode_state(query_params["config"])

    # --- セッション初期化 ---
    if 'assets' not in st.session_state:
        default_assets = [
            {'ticker': 'SPY', 'type': 'Long', 'allocation_pct': 50.0, 'margin_ratio': 100.0},
            {'ticker': 'TLT', 'type': 'Long', 'allocation_pct': 30.0, 'margin_ratio': 100.0},
        ]
        # デフォルト設定
        defaults = {
            'total_investment': 10000.0,
            'start_date': datetime.today() - timedelta(days=365),
            'risk_free_rate': 0.0,
            'rebalance_freq': 'Weekly'
        }

        if initial_config:
            st.session_state.total_investment = initial_config.get('total_investment', defaults['total_investment'])
            st.session_state.risk_free_rate = initial_config.get('risk_free_rate', defaults['risk_free_rate'])
            st.session_state.rebalance_freq = initial_config.get('rebalance_freq', defaults['rebalance_freq'])
            
            # Assetの復元（margin_ratio互換性対応）
            loaded_assets = initial_config.get('assets', default_assets)
            # Cash除外 & margin_ratioが無い場合は100(現物)を入れる
            clean_assets = []
            for a in loaded_assets:
                if a.get('type') == 'Cash': continue
                if 'margin_ratio' not in a: a['margin_ratio'] = 100.0
                clean_assets.append(a)
            st.session_state.assets = clean_assets
            
            saved_date_str = initial_config.get('start_date')
            if saved_date_str:
                try:
                    st.session_state.start_date = datetime.strptime(saved_date_str, '%Y-%m-%d').date()
                except ValueError:
                    st.session_state.start_date = defaults['start_date']
            else:
                st.session_state.start_date = defaults['start_date']
        else:
            st.session_state.total_investment = defaults['total_investment']
            st.session_state.assets = default_assets
            st.session_state.start_date = defaults['start_date']
            st.session_state.risk_free_rate = defaults['risk_free_rate']
            st.session_state.rebalance_freq = defaults['rebalance_freq']

    st.title("📈 Global Portfolio Backtester Pro")
    
    # --- サイドバー：全般設定 ---
    with st.sidebar:
        st.header("Global Settings")
        
        # 1. 投資額 & リスクフリーレート
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            total_inv = st.number_input("Initial Capital ($)", value=float(st.session_state.total_investment), step=1000.0, key='total_input')
            st.session_state.total_investment = total_inv
        with col_g2:
            rf_rate = st.number_input("Risk Free Rate (%)", value=float(st.session_state.risk_free_rate), step=0.1, key='rf_input')
            st.session_state.risk_free_rate = rf_rate

        # 2. 期間
        start_date_input = st.date_input("Start Date", value=st.session_state.start_date)
        st.session_state.start_date = start_date_input

        # 3. リバランス設定
        freq_options = ['None', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annually', 'Annually']
        try:
            freq_idx = freq_options.index(st.session_state.rebalance_freq)
        except ValueError:
            freq_idx = 2 # Default Weekly
            
        rebal_freq = st.selectbox("Rebalance Frequency", freq_options, index=freq_idx, key='rebal_input')
        st.session_state.rebalance_freq = rebal_freq

        st.divider()
        st.header("Asset Allocation")

        indices_to_remove = []
        updated_assets = []
        
        # 4. アセット入力
        for i, asset in enumerate(st.session_state.assets):
            with st.container(border=True):
                # ヘッダー行（タイトル + 削除ボタン）
                c_head1, c_head2 = st.columns([0.85, 0.15])
                with c_head1:
                    st.caption(f"Asset {i+1}")
                with c_head2:
                    if st.button("🗑️", key=f"del_{i}"):
                        indices_to_remove.append(i)

                # 行1: Ticker & Type
                c1, c2 = st.columns(2)
                with c1:
                    ticker = st.text_input("Ticker", value=asset['ticker'], key=f"tick_{i}", placeholder="SPY")
                with c2:
                    pos_type = st.selectbox("Type", ["Long", "Short"], index=["Long", "Short"].index(asset.get('type', 'Long')), key=f"type_{i}")
                
                # 行2: Allocation & Margin
                c3, c4 = st.columns(2)
                with c3:
                    current_pct = asset.get('allocation_pct', 0.0)
                    new_pct = st.number_input(f"Alloc (%)", value=float(current_pct), step=5.0, key=f"pct_{i}")
                with c4:
                    current_margin = asset.get('margin_ratio', 100.0)
                    new_margin = st.number_input(f"Margin (%)", value=float(current_margin), step=10.0, key=f"marg_{i}", help="証拠金率。現物買いなら100%、レバレッジならそれ以下を設定")
                
                # ガイダンス表示
                target_amt = total_inv * (new_pct/100)
                req_margin_amt = target_amt * (new_margin/100)
                if pos_type == 'Short':
                    st.caption(f"Short: +${target_amt:,.0f} | Margin Req: ${req_margin_amt:,.0f}")
                else:
                    st.caption(f"Long: -${target_amt:,.0f} | Margin Req: ${req_margin_amt:,.0f}")

                updated_assets.append({
                    'ticker': ticker.upper(), 
                    'type': pos_type, 
                    'allocation_pct': new_pct,
                    'margin_ratio': new_margin
                })

        if indices_to_remove:
            for index in sorted(indices_to_remove, reverse=True):
                updated_assets.pop(index)
            st.session_state.assets = updated_assets
            st.rerun()

        if st.button("➕ Add Asset"):
            updated_assets.append({'ticker': '', 'type': 'Long', 'allocation_pct': 0.0, 'margin_ratio': 100.0})
            st.session_state.assets = updated_assets
            st.rerun()

        st.session_state.assets = updated_assets
        st.divider()
        
        if st.button("💾 Save Config to URL"):
            config_to_save = {
                'total_investment': st.session_state.total_investment,
                'risk_free_rate': st.session_state.risk_free_rate,
                'rebalance_freq': st.session_state.rebalance_freq,
                'start_date': st.session_state.start_date,
                'assets': st.session_state.assets
            }
            encoded = encode_state(config_to_save)
            st.query_params["config"] = encoded
            st.success("Config saved to URL!")

    # --- メインエリア ---

    st.subheader("1. Portfolio Composition & Margin Check")
    
    if not st.session_state.assets:
        st.warning("Add assets to begin.")
        return

    df_assets = pd.DataFrame(st.session_state.assets)
    df_display = df_assets[df_assets['ticker'] != ''].copy()

    if df_display.empty:
        st.info("Enter tickers.")
        return

    # Cash計算
    long_pct = df_display[df_display['type'] == 'Long']['allocation_pct'].sum()
    short_pct = df_display[df_display['type'] == 'Short']['allocation_pct'].sum()
    calculated_cash_pct = 100.0 - long_pct + short_pct
    
    # 初期必要証拠金の計算
    initial_req_margin_pct = (df_display['allocation_pct'] * df_display['margin_ratio'] / 100.0).sum()
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Long", f"{long_pct:.1f}%")
    col_m2.metric("Total Short", f"{short_pct:.1f}%")
    col_m3.metric("Implied Cash", f"{calculated_cash_pct:.1f}%")
    col_m4.metric("Initial Margin Req", f"{initial_req_margin_pct:.1f}%", 
                  delta="OK" if initial_req_margin_pct <= 100 else "Over Leverage",
                  delta_color="normal" if initial_req_margin_pct <= 100 else "inverse")

    # 円グラフ
    pie_data = df_display[['ticker', 'allocation_pct', 'type']].copy()
    if calculated_cash_pct != 0:
        pie_data = pd.concat([pie_data, pd.DataFrame([{
            'ticker': 'CASH', 'allocation_pct': abs(calculated_cash_pct), 
            'type': 'Cash' if calculated_cash_pct > 0 else 'Debt'
        }])], ignore_index=True)

    col_pie1, col_pie2 = st.columns(2)
    with col_pie1:
        fig_pie = px.pie(pie_data, values='allocation_pct', names='ticker', hole=0.4, title="Asset Allocation")
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_pie2:
        df_type = pie_data.groupby('type')['allocation_pct'].sum().reset_index()
        fig_pie_type = px.pie(df_type, values='allocation_pct', names='type', color='type', title="Exposure Type",
                             color_discrete_map={'Long':'#00CC96', 'Short':'#EF553B', 'Cash':'#636EFA', 'Debt':'#AB63FA'})
        st.plotly_chart(fig_pie_type, use_container_width=True)

    # 2. バックテスト
    st.subheader(f"2. Historical Performance ({st.session_state.rebalance_freq} Rebalance)")
    
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

            # コンフィグ作成
            weights_config = {}
            margin_config = {}
            
            for _, row in df_display.iterrows():
                w = row['allocation_pct'] / 100.0
                if row['type'] == 'Short':
                    w = -1.0 * w
                weights_config[row['ticker']] = w
                margin_config[row['ticker']] = row['margin_ratio']
            
            # Cashの処理（ウェイトの残余）
            weights_config['CASH'] = calculated_cash_pct / 100.0

            # 計算用DF作成
            calc_df = prices_df.copy()
            if 'CASH' not in calc_df.columns:
                if calc_df.empty:
                     daterange = pd.date_range(start=st.session_state.start_date, end=datetime.today())
                     calc_df = pd.DataFrame(index=daterange)
                calc_df['CASH'] = 1.0
            else:
                calc_df['CASH'] = 1.0

            # シミュレーション実行
            equity_curve, margin_curve, daily_returns, rebalance_flags = run_backtest(
                calc_df, weights_config, total_inv, st.session_state.rebalance_freq, margin_config
            )

            # 指標計算
            metrics = calculate_metrics(daily_returns, risk_free_rate_pct=st.session_state.risk_free_rate)

            # 結果表示
            st.markdown("### 📊 Key Metrics")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            final_val = equity_curve.iloc[-1]
            total_ret_pct = (final_val / total_inv - 1) * 100
            
            m1.metric("Final Value", f"${final_val:,.0f}", f"{total_ret_pct:.1f}%")
            m2.metric("CAGR", f"{metrics['cagr']*100:.2f}%")
            m3.metric("Volatility", f"{metrics['volatility']*100:.2f}%")
            m4.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
            m5.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
            m6.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")

            # チャート作成
            fig = go.Figure()
            
            # 1. 資産推移
            fig.add_trace(go.Scatter(
                x=equity_curve.index, y=equity_curve.values, 
                mode='lines', name='Portfolio Equity', 
                line=dict(width=2, color='#636EFA')
            ))
            
            # 2. 必要証拠金（エリア）
            fig.add_trace(go.Scatter(
                x=margin_curve.index, y=margin_curve.values,
                mode='lines', name='Required Margin',
                fill='tozeroy', line=dict(width=1, color='rgba(239, 85, 59, 0.5)'),
                fillcolor='rgba(239, 85, 59, 0.1)'
            ))

            # 3. リバランスポイント（マーカー）
            rebal_dates = equity_curve.index[rebalance_flags]
            rebal_values = equity_curve[rebalance_flags]
            
            if len(rebal_dates) > 0:
                fig.add_trace(go.Scatter(
                    x=rebal_dates, y=rebal_values,
                    mode='markers', name='Rebalance Event',
                    marker=dict(symbol='diamond', size=8, color='gold', line=dict(width=1, color='black'))
                ))

            fig.update_layout(
                title=f'Portfolio Value & Margin Requirement ({st.session_state.rebalance_freq})', 
                xaxis_title='Date', yaxis_title='Value ($)', 
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # 個別銘柄パフォーマンス
            st.subheader("3. Asset Performance (Base=100)")
            if not prices_df.empty:
                disp_cols = [c for c in prices_df.columns if c != 'CASH']
                if disp_cols:
                    normalized_df = prices_df[disp_cols] / prices_df[disp_cols].iloc[0] * 100
                    fig_ind = px.line(normalized_df, title="Asset Price Movement")
                    st.plotly_chart(fig_ind, use_container_width=True)

if __name__ == "__main__":
    main()
