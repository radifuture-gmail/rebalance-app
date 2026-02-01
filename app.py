import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ページ設定
st.set_page_config(page_title="GDX/UGL Pair Trader", layout="wide")

st.title("📊 GDX/UGL ペアトレード Zスコア判定")

# サイドバーで設定変更可能にする
with st.sidebar:
    st.header("設定")
    window = st.slider("移動平均期間 (日)", 50, 300, 120, help="200日は安定的ですが、相場の構造変化への対応が遅れるリスクがあります。")
    z_threshold = st.slider("シグナル閾値 (Z)", 1.0, 3.0, 1.0)
    st.markdown("---")
    st.markdown("**戦略:**\n\nZスコアに応じてポジション調整。\n乖離が異常値(>2.0)の場合はキャップを適用。")

# データ取得関数（キャッシュして高速化）
@st.cache_data
def get_data():
    tickers = ['UGL', 'GDX']
    # 過去2年分取得（移動平均計算のため長めに）
    df = yf.download(tickers, period='2y', auto_adjust=True)
    
    # マルチインデックスの場合の処理（yfinanceのバージョンによる差異吸収）
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df['Close']
        except KeyError:
            pass # 既にCloseのみの場合
            
    return df

try:
    with st.spinner('データを取得中...'):
        df = get_data()
        st.write(f"取得データ数: {len(df)} 行") # ここで500前後なら正常

    if df.empty:
        st.error("データの取得に失敗しました。時間をおいて再試行してください。")
        st.stop()

    # 指標計算
    # 対数比率: ln(UGL) - ln(GDX)
    df['Log_Ratio'] = np.log(df['UGL']) - np.log(df['GDX'])
    
    # 移動平均と標準偏差
    df['Mean'] = df['Log_Ratio'].rolling(window=window).mean()
    df['Std'] = df['Log_Ratio'].rolling(window=window).std()
    
    # Zスコア
    df['Z_Score'] = (df['Log_Ratio'] - df['Mean']) / df['Std']

    # 最新データ
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    current_z = latest['Z_Score']
    current_ratio = latest['Log_Ratio']
    
    # --- メイン画面表示 ---

    # 1. 判定結果を大きく表示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("現在のZスコア", f"{current_z:.2f}", delta=f"{current_z - prev['Z_Score']:.2f}")

    with col2:
        # アクション判定
        if current_z < -z_threshold:
            status = "🔥 買い増し (UGL割安)"
            color = "red"
            instruction = "ポジションを+0.1積み増し（キャップ注意）"
        elif current_z > z_threshold:
            status = "💰 利益確定 (UGL割高)"
            color = "green"
            instruction = "ポジションを縮小"
        else:
            status = "☕ 待機 (範囲内)"
            color = "gray"
            instruction = "現状維持"
            
        st.subheader(f"判定: :{color}[{status}]")
        st.caption(instruction)

    with col3:
        st.metric("UGL価格", f"${latest['UGL']:.2f}")
        st.metric("GDX価格", f"${latest['GDX']:.2f}")

    # 2. チャート表示 (Plotly)
    st.subheader("Zスコアの推移")
    
    # データを直近2年分に絞る
    plot_df = df.iloc[-504:]
    
    fig = go.Figure()
    
    # Zスコア線
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Z_Score'], mode='lines', name='Z-Score', line=dict(color='blue')))
    
    # 閾値ライン
    fig.add_hline(y=z_threshold, line_dash="dash", line_color="green", annotation_text="売りゾーン")
    fig.add_hline(y=-z_threshold, line_dash="dash", line_color="red", annotation_text="買いゾーン")
    fig.add_hline(y=0, line_color="gray", opacity=0.5)

    fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 3. 生データの確認（オプション）
    with st.expander("詳細データを見る"):
        st.dataframe(df.tail(365).sort_index(ascending=False))

except Exception as e:
    st.error(f"エラーが発生しました: {e}")

# --- リバランス計算機 ---
st.markdown("---")
st.header("🧮 リバランス計算機")

with st.form("rebalance_form"):
    # ユーザー入力
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        current_nlv = st.number_input("目標ポジション総額 (USD)", value=48000.0, step=100.0)
        current_ugl_val = st.number_input("現在のUGL保有額 (USD)", value=24000.0, step=100.0)
    with col_calc2:
        # 現在のポジション調整倍率（シグナルに基づく推奨値）
        # ロジック: 基本1.0 + (Zスコアに応じた調整)
        # 簡易ロジック例: Z < -1 なら 1.1, Z > 1 なら 0.9, それ以外 1.0
        rec_scale = 1.0
        if current_z < -1.0: rec_scale = 1.1
        if current_z < -2.0: rec_scale = 1.2 # キャップ
        if current_z > 1.0: rec_scale = 0.9
        if current_z > 2.0: rec_scale = 0.7
        
        target_scale = st.number_input("ターゲット・ポジション倍率", value=rec_scale, step=0.1, help="1.0=資産と同額, 1.1=10%レバレッジ")
        current_gdx_val = st.number_input("現在のGDX空売り額 (USD)", value=24000.0, step=100.0)

    submitted = st.form_submit_button("計算する")

    if submitted:
        # 目標金額
        total_target = current_nlv * target_scale
        target_ugl_val = total_target / 2
        target_gdx_val = total_target / 2 # 金額ベース（ショートなので絶対値）

        # 差額計算
        diff_ugl = target_ugl_val - current_ugl_val
        diff_gdx = target_gdx_val - current_gdx_val
        
        # 株数換算
        qty_ugl = int(diff_ugl / latest['UGL'])
        qty_gdx = int(diff_gdx / latest['GDX']) # プラスなら売り増し、マイナスなら買い戻し

        st.info(f"目標ポジション総額: ${total_target:,.2f} (各 ${target_ugl_val:,.2f})")
        
        c1, c2 = st.columns(2)
        with c1:
            if qty_ugl > 0:
                st.success(f"UGL: {qty_ugl} 株 **買い** (${diff_ugl:,.2f})")
            elif qty_ugl < 0:
                st.warning(f"UGL: {abs(qty_ugl)} 株 **売り** (-${abs(diff_ugl):,.2f})")
            else:
                st.write("UGL: 売買なし")
        
        with c2:
            # GDXはショートポジションの話なので表現に注意
            # 保有額(Short Value)を増やしたい = 新規空売り
            # 保有額(Short Value)を減らしたい = 買い戻し
            if qty_gdx > 0:
                st.error(f"GDX: {qty_gdx} 株 **新規空売り** (ポジション増)")
            elif qty_gdx < 0:
                st.success(f"GDX: {abs(qty_gdx)} 株 **買い戻し** (ポジション減)")
            else:
                st.write("GDX: 売買なし")