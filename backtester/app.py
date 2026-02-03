import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots  # 追加: サブプロット用
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
        # yfinanceの仕様変更等に対応するため、シンプルな取得方法を維持
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
    """
    if prices.empty:
        return None, None, None, None

    prices.index = pd.to_datetime(prices.index)
    
    dates = prices.index
    portfolio_values = np.zeros(len(dates))
    required_margins = np.zeros(len(dates))
    rebalance_flags = np.zeros(len(dates), dtype=bool)
    
    current_shares = {ticker: 0.0 for ticker in weights_config.keys()}
    cash_holdings = initial_capital 

    # リバランス日の判定
    if rebalance_freq == 'None':
        rb_dates = [dates[0]]
    else:
        freq_map = {
            'Daily': 'D', 'Weekly': 'W-FRI', 'Monthly': 'ME', 
            'Quarterly': 'QE', 'Semi-Annually': '6ME', 'Annually': 'YE'
        }
        freq_str = freq_map.get(rebalance_freq, 'D')
        
        rb_dates_idx = pd.date_range(start=dates[0], end=dates[-1], freq=freq_str)
        
        rb_dates = []
        rb_dates.append(dates[0]) # 初日は必ず
        
        for target_date in rb_dates_idx:
            if target_date <= dates[0]: continue
            future_dates = dates[dates >= target_date]
            if not future_dates.empty:
                next_date = future_dates[0]
                if next_date not in rb_dates:
                    rb_dates.append(next_date)
    
    rb_dates_set = set(rb_dates)

    # 日次ループ
    for i, date in enumerate(dates):
        price_row = prices.iloc[i]
        
        # 1. リバランス判定 & 実行
        if date in rb_dates_set:
            rebalance_flags[i] = True
            
            # 現在の総資産額
            current_equity = cash_holdings
            for ticker, shares in current_shares.items():
                if ticker == 'CASH': continue
                current_equity += shares * price_row[ticker]
            
            # 再配分
            new_shares = {}
            new_cash = current_equity
            
            for ticker, target_w in weights_config.items():
                if ticker == 'CASH': continue
                
                target_amt = current_equity * target_w
                price = price_row[ticker]
                
                if price != 0:
                    shares = target_amt / price
                    new_shares[ticker] = shares
                    new_cash -= target_amt
            
            current_shares = new_shares
            cash_holdings = new_cash
        
        # 2. 評価額計算
        daily_equity = cash_holdings
        daily_margin_req = 0.0
        
        for ticker, shares in current_shares.items():
            if ticker == 'CASH': continue
            
            val = shares * price_row[ticker]
            daily_equity += val
            
            m_ratio = margin_ratios.get(ticker, 0.0)
            daily_margin_req += abs(val) * (m_ratio / 100.0)
            
        portfolio_values[i] = daily_equity
        required_margins[i] = daily_margin_req

    portfolio_series = pd.Series(portfolio_values, index=dates)
    margin_series = pd.Series(required_margins, index=dates)
    daily_returns = portfolio_series.pct_change().fillna(0)
    
    return portfolio_series, margin_series, daily_returns, rebalance_flags

def calculate_metrics(daily_ret, risk_free_rate_pct=0.0):
    """各種指標計算"""
    ann_factor = 252
    rf = risk_free_rate_pct / 100.0
    
    total_ret = (1 + daily_ret).prod() - 1
    n_years = len(daily_ret) / ann_factor
    if n_years > 0:
        cagr = (1 + total_ret) ** (1/n_years) - 1
    else:
        cagr = 0.0

    volatility = daily_ret.std() * np.sqrt(ann_factor)
    
    sharpe = 0.0
    if volatility != 0:
        sharpe = (cagr - rf) / volatility
        
    cumulative = (1 + daily_ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
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

    # --- URLパラメータ ---
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
            
            loaded_assets = initial_config.get('assets', default_assets)
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
    
    # --- サイドバー ---
    with st.sidebar:
        st.header("Global Settings")
        
        c1, c2 = st.columns(2)
        with c1:
            total_inv = st.number_input("Initial Capital ($)", value=float(st.session_state.total_investment), step=1000.0, key='total_input')
            st.session_state.total_investment = total_inv
        with c2:
            rf_rate = st.number_input("Risk Free Rate (%)", value=float(st.session_state.risk_free_rate), step=0.1, key='rf_input')
            st.session_state.risk_free_rate = rf_rate

        start_date_input = st.date_input("Start Date", value=st.session_state.start_date)
        st.session_state.start_date = start_date_input

        freq_options = ['None', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Semi-Annually', 'Annually']
        try:
            freq_idx = freq_options.index(st.session_state.rebalance_freq)
        except ValueError:
            freq_idx = 2 
            
        rebal_freq = st.selectbox("Rebalance Frequency", freq_options, index=freq_idx, key='rebal_input')
        st.session_state.rebalance_freq = rebal_freq

        st.divider()
        st.header("Asset Allocation")

        indices_to_remove = []
        updated_assets = []
        
        for i, asset in enumerate(st.session_state.assets):
            with st.container(border=True):
                c_head1, c_head2 = st.columns([0.85, 0.15])
                with c_head1:
                    st.caption(f"Asset {i+1}")
                with c_head2:
                    if st.button("🗑️", key=f"del_{i}"):
                        indices_to_remove.append(i)

                c1, c2 = st.columns(2)
                with c1:
                    ticker = st.text_input("Ticker", value=asset['ticker'], key=f"tick_{i}", placeholder="SPY")
                with c2:
                    pos_type = st.selectbox("Type", ["Long", "Short"], index=["Long", "Short"].index(asset.get('type', 'Long')), key=f"type_{i}")
                
                c3, c4 = st.columns(2)
                with c3:
                    current_pct = asset.get('allocation_pct', 0.0)
                    new_pct = st.number_input(f"Alloc (%)", value=float(current_pct), step=5.0, key=f"pct_{i}")
                with c4:
                    current_margin = asset.get('margin_ratio', 100.0)
                    new_margin = st.number_input(f"Margin (%)", value=float(current_margin), step=10.0, key=f"marg_{i}")
                
                target_amt = total_inv * (new_pct/100)
                req_margin_amt = target_amt * (new_margin/100)
                if pos_type == 'Short':
                    st.caption(f"Short: +${target_amt:,.0f} | Margin: ${req_margin_amt:,.0f}")
                else:
                    st.caption(f"Long: -${target_amt:,.0f} | Margin: ${req_margin_amt:,.0f}")

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

    long_pct = df_display[df_display['type'] == 'Long']['allocation_pct'].sum()
    short_pct = df_display[df_display['type'] == 'Short']['allocation_pct'].sum()
    calculated_cash_pct = 100.0 - long_pct + short_pct
    initial_req_margin_pct = (df_display['allocation_pct'] * df_display['margin_ratio'] / 100.0).sum()
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Long", f"{long_pct:.1f}%")
    col_m2.metric("Total Short", f"{short_pct:.1f}%")
    col_m3.metric("Implied Cash", f"{calculated_cash_pct:.1f}%")
    col_m4.metric("Initial Margin Req", f"{initial_req_margin_pct:.1f}%", 
                  delta="OK" if initial_req_margin_pct <= 100 else "Over Leverage",
                  delta_color="normal" if initial_req_margin_pct <= 100 else "inverse")

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

            weights_config = {}
            margin_config = {}
            for _, row in df_display.iterrows():
                w = row['allocation_pct'] / 100.0
                if row['type'] == 'Short': w = -1.0 * w
                weights_config[row['ticker']] = w
                margin_config[row['ticker']] = row['margin_ratio']
            weights_config['CASH'] = calculated_cash_pct / 100.0

            calc_df = prices_df.copy()
            if 'CASH' not in calc_df.columns:
                if calc_df.empty:
                     daterange = pd.date_range(start=st.session_state.start_date, end=datetime.today())
                     calc_df = pd.DataFrame(index=daterange)
                calc_df['CASH'] = 1.0
            else:
                calc_df['CASH'] = 1.0

            # 計算実行
            equity_curve, margin_curve, daily_returns, rebalance_flags = run_backtest(
                calc_df, weights_config, total_inv, st.session_state.rebalance_freq, margin_config
            )

            metrics = calculate_metrics(daily_returns, risk_free_rate_pct=st.session_state.risk_free_rate)

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

            # --- チャート描画（サブプロット使用） ---
            # Row 1: Equity (80%), Row 2: Rebalance Points (20%)
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.03, 
                row_heights=[0.8, 0.2],
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            # Row 1: Equity & Margin
            fig.add_trace(go.Scatter(
                x=equity_curve.index, y=equity_curve.values, 
                mode='lines', name='Portfolio Equity', 
                line=dict(width=2, color='#636EFA')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=margin_curve.index, y=margin_curve.values,
                mode='lines', name='Required Margin',
                fill='tozeroy', line=dict(width=1, color='rgba(239, 85, 59, 0.5)'),
                fillcolor='rgba(239, 85, 59, 0.1)'
            ), row=1, col=1)

            # Row 2: Rebalance Markers
            rebal_dates = equity_curve.index[rebalance_flags]
            if len(rebal_dates) > 0:
                fig.add_trace(go.Scatter(
                    x=rebal_dates, 
                    # y=0 に一直線にプロット
                    y=[0] * len(rebal_dates),
                    mode='markers', name='Rebalance Event',
                    marker=dict(symbol='diamond', size=10, color='gold', line=dict(width=1, color='black'))
                ), row=2, col=1)

            # Layout設定
            fig.update_layout(
                title=f'Portfolio Value & Rebalance Events ({st.session_state.rebalance_freq})', 
                hovermode="x unified",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # 軸ラベル調整
            fig.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Event", showticklabels=False, row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("3. Asset Performance (Base=100)")
            if not prices_df.empty:
                disp_cols = [c for c in prices_df.columns if c != 'CASH']
                if disp_cols:
                    normalized_df = prices_df[disp_cols] / prices_df[disp_cols].iloc[0] * 100
                    fig_ind = px.line(normalized_df, title="Asset Price Movement")
                    st.plotly_chart(fig_ind, use_container_width=True)

if __name__ == "__main__":
    main()
