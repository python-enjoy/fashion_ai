
# 必要なライブラリをインポート
import streamlit as st
from PIL import Image, ImageEnhance
from prediction import predict  # モデルを利用して画像を予測する関数をインポート

# Webアプリの外観をカスタマイズするためのCSSを設定
st.markdown("""
<style>
h1 {
    color: #6B8E23;
}
</style>
    """, unsafe_allow_html=True)

# Webアプリのタイトルを設定
st.title("ファッション画像認識アプリ")
st.write("服やカバン等のファッションアイテム画像をオリジナルAIが判定します")

# サイドバーに、画像の取得方法を選択するためのラジオボタンを設置
image_source_option = st.sidebar.radio("画像データを選択", ("画像ファイルを選択", "カメラでリアルタイム撮影"))

uploaded_image = None  # 画像データを初期化

# 選択された画像の取得方法に基づいて動作を変更
if image_source_option == "画像ファイルを選択":
    uploaded_image = st.sidebar.file_uploader("画像を選択してください。", type=["png", "jpg", "jpeg"])
elif image_source_option == "カメラでリアルタイム撮影":
    uploaded_image = st.sidebar.camera_input("カメラでリアルタイム撮影")

# サイドバーに画像の明るさとコントラストを調整するスライダーを追加
brightness_level = st.sidebar.slider("明るさ調整", 0.5, 1.5, 1.0)
contrast_level = st.sidebar.slider("コントラスト調整", 0.5, 1.5, 1.0)

# セッションステートを使用して、正解・不正解のカウントを保持（初回のみ初期化）
if 'correct_count' not in st.session_state:
    st.session_state.correct_count = 0
if 'incorrect_count' not in st.session_state:
    st.session_state.incorrect_count = 0

# 画像がアップロードされたら
if uploaded_image is not None:
    # PILライブラリを使って、アップロードされた画像を開く
    display_image = Image.open(uploaded_image)
    # 選択された明るさとコントラストのレベルで画像を調整
    display_image = ImageEnhance.Brightness(display_image).enhance(brightness_level)
    display_image = ImageEnhance.Contrast(display_image).enhance(contrast_level)

    # 画像をWebアプリに表示
    st.image(display_image, caption="対象のファッションアイテム画像", use_column_width=True)

    # 画像をモデルに入力して、認識を行う
    with st.spinner("推定しています..."):
        predictions = predict(display_image)  # 画像認識関数を呼び出し

        # 認識結果を表示
        st.subheader("このアイテムの判定結果")
        primary_prediction = predictions[0]
        st.write(f"{primary_prediction[2]*100:.2f}%の確率で{primary_prediction[0]}です。")

        # フィードバックボタンを配置
        correct_feedback = st.button("認識が合っている")
        incorrect_feedback = st.button("間違っている")

        # ボタンのフィードバックを元にカウンタを更新
        if correct_feedback:
            st.write("ありがとう！フィードバックを受け取りました。")
            st.session_state.correct_count += 1  # 正解のカウントを増やす
        elif incorrect_feedback:
            st.write("フィードバックを受け取りました。改善のために頑張ります！")
            st.session_state.incorrect_count += 1  # 不正解のカウントを増やす

        # 全フィードバックからの正解率を計算して表示
        total_feedbacks = st.session_state.correct_count + st.session_state.incorrect_count
        if total_feedbacks > 0:
            accuracy_percentage = (st.session_state.correct_count / total_feedbacks) * 100
            st.write(f"正解率: {accuracy_percentage:.2f}%")

# サイドバーの下部に著作権情報等を追記
st.sidebar.caption("""
Learned Fashion-MNIST\n
https://github.com/zalandoresearch/fashion-mnist#license\n
Copyright (c) 2017 Zalando SE\n
Released under the MIT license\n
https://opensource.org/licenses/mit-license.php
""")

