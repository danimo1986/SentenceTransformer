import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# データの読み込み
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    column_name = 'keyword'

    # SentenceTransformerモデルの初期化
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # テキストをベクトル化
    corpus_list = df[column_name].to_list()
    corpus_embeddings = model.encode(corpus_list, convert_to_tensor=True)  # ここで定義

    # プログレスバーを初期化
    progress_bar = st.progress(0)

    # キーワード入力
    keywords = st.text_input("キーワードを入力してください（複数のキーワードはスペースで区切ってください）")

    if keywords:
        keywords = keywords.split()
        keyword_sum = '|'.join(keywords)
        keyword_sum_embedding = model.encode(keyword_sum, convert_to_tensor=True)

        # コサイン類似度を計算
        cos_scores = util.cos_sim(keyword_sum_embedding, corpus_embeddings)[0]
        number_of_texts_shown = st.slider("表示するテキストの数", min_value=3, max_value=10, step=1, value=5)

        # 検索結果を表示
        st.markdown('### Keywords と意味が近いテキストを検索した結果')
        st.write("Keyword:", keywords, "のAnd検索")

        top_k = min(number_of_texts_shown, len(corpus_list))
        
        # top_resultsを定義
        top_results = torch.topk(cos_scores, k=top_k)
        
        # 進捗バーを表示
        progress_bar_results = st.progress(0)

        for i, idx in enumerate(top_results[1]):  # 修正: インデックスを正しく使用
            score = cos_scores[idx]
            st.write('✓', corpus_list[idx], "(Cosine Similarity: {:.3f})".format(score))
            # バッチごとに進捗バーを更新
            progress = (i + 1) / top_k
            progress_bar_results.progress(progress)


