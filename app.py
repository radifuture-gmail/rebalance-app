import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import json

# ページ設定
st.set_page_config(page_title="GDX/UGL Pair Trader", layout="wide")

# --- 1. URLパラメータの読み込みとデコード処理 ---
# デフォルト値を定義
default_values = {
    "window": 120,
    "z_threshold": 1.0,
    "nlv": 48000.0,
    "ugl_val": 24000.0,
    "gdx_val": 24000.0
}

# URLパラメータを取得
query_params = st.query_params
loaded_state = default_values.copy()

if "q" in query_params:
    try:
        # BASE64文字列をデコード -> JSONパース
        b64_str = query_params["q"]
        json_str = base64.b64decode(b64_str).decode('utf-8')
        loaded_state.update(json.loads(json_str))
        st.toast("URLから設定を読み込みました", icon="✅")
    except Exception:
        st.error("URLパラメータの読み込みに失敗しました。デフォルト値を使用します。")

st.title("📊 GDX/UGL ペアトレード Zスコア判定")

# --- サイドバー設定 ---
with st.sidebar:
    st.header("設定")
    # valueにloaded_state['...'] を指定することで、URLからの値を反映
    window = st.slider("移動平均期間 (日)", 50, 300, 
                       value=int(loaded_state['window']), 
                       help="200日は安定的ですが、相場の構造変化への対応が遅れるリスクがあります。")
    
    z_threshold = st.slider("シグナル閾値 (Z)", 1.0, 3.0, 
                            value=float(loaded_state['z_threshold']))
    
    st.markdown("---")
    st.markdown("**戦略:**\n\nZスコアに応じてポジション調整。\n乖離が異常値(>2.0)の場合はキャップを適用。")

# データ取得関数（キャッシュ設定: 1時間で更新）
@st.cache_data(ttl=3600)
def get_data():
    tickers = ['UGL', 'GDX']
    df = yf.download(tickers, period='2y', auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df['Close']
        except KeyError:
            pass
    return df

try:
    with st.spinner('データを取得中...'):
        df = get_data()

    if df.empty:
        st.error("データの取得に失敗しました。")
        st.stop()

    # 指標計算
    df['Log_Ratio'] = np.log(df['UGL']) - np.log(df['GDX'])
    df['Mean'] = df['Log_Ratio'].rolling(window=window).mean()
    df['Std'] = df['Log_Ratio'].rolling(window=window).std()
    df['Z_Score'] = (df['Log_Ratio'] - df['Mean']) / df['Std']

    # 最新データ
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    current_z = latest['Z_Score']
    
    # --- メイン画面表示 (判定ロジックなど) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("現在のZスコア", f"{current_z:.2f}", delta=f"{current_z - prev['Z_Score']:.2f}")

    with col2:
        if current_z < -z_threshold:
            status, color, instruction = "🔥 買い増し (UGL割安)", "red", "ポジションを+0.1積み増し"
        elif current_z > z_threshold:
            status, color, instruction = "💰 利益確定 (UGL割高)", "green", "ポジションを縮小"
        else:
            status, color, instruction = "☕ 待機 (範囲内)", "gray", "現状維持"
            
        st.subheader(f"判定: :{color}[{status}]")
        st.caption(instruction)

    with col3:
        st.metric("UGL価格", f"${latest['UGL']:.2f}")
        st.metric("GDX価格", f"${latest['GDX']:.2f}")

    # チャート表示
    st.subheader("Zスコアの推移")
    plot_df = df.iloc[-504:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Z_Score'], mode='lines', name='Z-Score', line=dict(color='blue')))
    fig.add_hline(y=z_threshold, line_dash="dash", line_color="green")
    fig.add_hline(y=-z_threshold, line_dash="dash", line_color="red")
    fig.add_hline(y=0, line_color="gray", opacity=0.5)
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"エラーが発生しました: {e}")

# --- リバランス計算機 ---
st.markdown("---")
st.header("🧮 リバランス計算機")

# 入力フォームの外側で値を保持するためのコンテナ
with st.container():
    # ユーザー入力 (valueに loaded_state を使用)
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        current_nlv = st.number_input("目標ポジション総額 (USD)", 
                                      value=float(loaded_state['nlv']), step=100.0)
        current_ugl_val = st.number_input("現在のUGL保有額 (USD)", 
                                          value=float(loaded_state['ugl_val']), step=100.0)
    with col_calc2:
        # 現在のポジション調整倍率計算
        rec_scale = 1.0
        if current_z < -1.0: rec_scale = 1.1
        if current_z < -2.0: rec_scale = 1.2
        if current_z > 1.0: rec_scale = 0.9
        if current_z > 2.0: rec_scale = 0.7
        if current_z > 3.0: rec_scale = 0.5
        
        target_scale = st.number_input("ターゲット・ポジション倍率", value=rec_scale, step=0.1)
        current_gdx_val = st.number_input("現在のGDX空売り額 (USD)", 
                                          value=float(loaded_state['gdx_val']), step=100.0)

    # 計算実行ボタン
    if st.button("計算する"):
        total_target = current_nlv * target_scale
        target_ugl_val = total_target / 2
        target_gdx_val = total_target / 2

        diff_ugl = target_ugl_val - current_ugl_val
        diff_gdx = target_gdx_val - current_gdx_val
        
        try:
            qty_ugl = int(diff_ugl / latest['UGL'])
            qty_gdx = int(diff_gdx / latest['GDX'])
        except:
            qty_ugl = 0; qty_gdx = 0

        st.info(f"目標ポジション総額: ${total_target:,.2f}")
        
        c1, c2 = st.columns(2)
        with c1:
            if qty_ugl > 0: st.success(f"UGL: {qty_ugl} 株 **買い** (${diff_ugl:,.2f})")
            elif qty_ugl < 0: st.warning(f"UGL: {abs(qty_ugl)} 株 **売り** (-${abs(diff_ugl):,.2f})")
            else: st.write("UGL: 売買なし")
        
        with c2:
            if qty_gdx > 0: st.error(f"GDX: {qty_gdx} 株 **新規空売り** (ポジション増)")
            elif qty_gdx < 0: st.success(f"GDX: {abs(qty_gdx)} 株 **買い戻し** (ポジション減)")
            else: st.write("GDX: 売買なし")

# --- 2. 設定保存ボタンの実装 ---
st.markdown("---")
st.subheader("💾 設定の共有・保存")

if st.button("現在の入力値をURLに保存"):
    # 現在のステートを辞書にまとめる
    current_state = {
        "window": window,
        "z_threshold": z_threshold,
        "nlv": current_nlv,
        "ugl_val": current_ugl_val,
        "gdx_val": current_gdx_val
    }
    
    # JSON文字列化 -> Bytes -> BASE64エンコード -> 文字列化
    json_str = json.dumps(current_state)
    b64_str = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    
    # URLパラメータを更新
    st.query_params["q"] = b64_str
    
    # ユーザーにURLを提示
    st.success("URLを生成しました！ブラウザのアドレスバーが更新されました。ブックマークするか、下記をコピーしてください。")
    # 現在のベースURLを取得するのは環境によるため、相対パス的なパラメータ表示に留めるか、単純に再読み込みを促す
    st.code(f"?q={b64_str}", language="text")
    st.info("👆 このパラメータを含むURLにアクセスすると、現在の入力値が復元されます。")
