import re
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

import nltk
import numpy as np
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
"""
自然言語処理においては、出現頻度が高いにも関わらず特別な意味をもたない不要な単語を削除する必要があります。
この不要な語をストップワードと言います。
日本語で言えば、「は」、「です」、「する」などがストップワードに相当します。
英語で言えば「i」, 「a」, 「is」などがストップワードに相当します。
"""


class TfidfPreprocessor:
    @staticmethod
    def preprocess_TAGS(x: pd.Series) -> pd.Series:
        """TAGS列の前処理をする関数

        Args:
            x (pd.Series): _description_

        Returns:
            pd.Series: _description_
        """
        x = x.split(",")
        x = [t.strip() for t in x]
        x = [t.replace("#", "") for t in x]
        x = " ".join(x)
        return x

    @staticmethod
    def preprocess_LOANUSE(x: pd.Series):
        """LOAN_USE列の前処理をする関数

        Args:
            x (pd.Series): _description_

        Returns:
            _type_: _description_
        """
        x = re.sub(r"[^a-zA-Z0-9\s]", " ", x)  # 英数字とスペース以外の文字をスペースに変換する
        x_tokenized = word_tokenize(x)  # スペースは削除し単語だけにする

        word_list = []
        stem = PorterStemmer()  # 活用された単語を語幹、基本、またはルート形式（一般的には書き言葉の形式）に変換するプロセス
        STOP_WORDS = set(stopwords.words("english"))  # 出現頻度が高いにも関わらず特別な意味をもたない不要な単語
        for word in x_tokenized:
            if word not in STOP_WORDS:
                word = stem.stem(word)
                word_list.append(word)

        x_stemmed = " ".join(word_list)
        return x_stemmed


class TfidfModule:
    def __init__(self, target):
        self.__target = target

    def fit_transform(self, dataframe: pd.DataFrame):
        """tf_idf計算し、SVD(特異値分解)して２次元の特徴量にする関数
        TF: 文書中により高頻度で出現する単語ほど，その文書にとって重要だ
        IDF: 特定の文書に出現する単語ほど，ある話題に特化した意味のある単語である

        TfidfVectorizer:
            sublinear_tf: tfの値を+1して対数で計算するかどうか

        Args:
            dataframe (pd.DataFrame): _description_

        Returns:
            _type_: _description_
        """
        df = dataframe[["LOAN_ID", self.__target]].copy()

        ins_prep = TfidfPreprocessor()
        if self.__target == "TAGS":
            df[f"fixed_{self.__target}"] = df[self.__target].astype(str).apply(lambda x: ins_prep.preprocess_TAGS(x))
        elif self.__target == "LOAN_USE":
            df[f"fixed_{self.__target}"] = df[self.__target].astype(str).apply(lambda x: ins_prep.preprocess_LOANUSE(x))

        # Tf-idf
        tfidf_vectorizer = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, use_idf=True, smooth_idf=True)
        tfidf_vectorizer.fit(df[f"fixed_{self.__target}"])
        tfidf_results = tfidf_vectorizer.transform(df[f"fixed_{self.__target}"])
        # array = tfidf_results.toarray()  # array型にも変換できる

        # SVD
        svd = decomposition.TruncatedSVD(n_components=2, random_state=0)
        svd.fit(tfidf_results)
        df_tfidf_results = pd.DataFrame(svd.transform(tfidf_results))
        df_tfidf_results.columns = [f"TFIDF_1_{self.__target}", f"TFIDF_2_{self.__target}"]

        # concat
        df_output = pd.concat([df, df_tfidf_results], axis=1)
        return tfidf_vectorizer, svd, df_output

    def transform(self, tfidf_vectorizer, svd, dataframe: pd.DataFrame):
        df = dataframe[["LOAN_ID", self.__target]].copy()

        ins_prep = TfidfPreprocessor()
        if self.__target == "TAGS":
            df[f"fixed_{self.__target}"] = df[self.__target].astype(str).apply(lambda x: ins_prep.preprocess_TAGS(x))
        elif self.__target == "LOAN_USE":
            df[f"fixed_{self.__target}"] = df[self.__target].astype(str).apply(lambda x: ins_prep.preprocess_LOANUSE(x))

        tfidf_results = tfidf_vectorizer.transform(df[f"fixed_{self.__target}"])
        df_tfidf_results = pd.DataFrame(svd.transform(tfidf_results))
        df_tfidf_results.columns = [f"TFIDF_1_{self.__target}", f"TFIDF_2_{self.__target}"]

        # concat
        df_output = pd.concat([df, df_tfidf_results], axis=1)
        return df_output
