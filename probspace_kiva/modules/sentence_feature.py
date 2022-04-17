import re
import string
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import nltk
import pandas as pd
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize

"""Downloading package punkt
"""
nltk.download('punkt')


class CleanSentences:
    def __init__(self, config: dict):
        self.primary_key = list(config.primary_key.keys())
        self.sentence_columns = config.sentence_columns

    def execute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe[self.primary_key + self.sentence_columns].copy()
        for s in self.sentence_columns:
            df[s] = df[s].apply(lambda x: self._clean_html_tags(x))
            df[s] = df[s].apply(lambda x: self._clean_empty_expressions(x))
            df[s] = df[s].apply(lambda x: self._clean_spaces(x))
            df[s] = df[s].apply(lambda x: self._remove_urls(x))
            df[s] = df[s].apply(lambda x: self._clean_puncts(x))
        return df

    def _change_upper_to_lower(self, x: pd.Series) -> pd.Series:
        """大文字を小文字に変更する関数

        Args:
            x (pd.Series): _description_

        Returns:
            pd.Series: _description_
        """
        return str(x).lower()

    def _clean_html_tags(self, x: pd.Series) -> pd.Series:
        html_tags = [
            '<p>', '</p>', '<table>', '</table>', '<tr>', '</tr>', '<ul>', '<ol>', '<dl>',
            '</ul>', '</ol>', '</dl>', '<li>', '<dd>', '<dt>', '</li>', '</dd>', '</dt>', '<h1>', '</h1>',
            '<br>', '<br/>', '<br />', '<strong>', '</strong>', '<span>', '</span>', '<blockquote>', '</blockquote>',
            '<pre>', '</pre>', '<div>', '</div>', '<h2>', '</h2>', '<h3>', '</h3>', '<h4>', '</h4>', '<h5>', '</h5>',
            '<h6>', '</h6>', '<blck>', '<pr>', '<code>', '<th>', '</th>', '<td>', '</td>', '<em>', '</em>'
        ]
        for h in html_tags:
            x = str(x).replace(h, "")
        return x

    def _clean_empty_expressions(self, x: pd.Series) -> pd.Series:
        empty_expressions = [
            '&lt;', '&gt;', '&amp;', '&nbsp;',
            '&emsp;', '&ndash;', '&mdash;', '&ensp;', '&quot;', '&#39;'
        ]
        for e in empty_expressions:
            x = x.replace(e, "")
        return x

    def _clean_spaces(self, x: pd.Series) -> pd.Series:
        spaces = [
            '\u200b', '\u200e', '\u202a', '\u2009', '\u2028', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u3000',
            '\x10', '\x7f', '\x9d', '\xad', '\x97', '\x9c', '\x8b', '\x81', '\x80', '\x8c', '\x85', '\x92', '\x88',
            '\x8d', '\x80', '\x8e', '\x9a', '\x94', '\xa0', '\x8f', '\x82', '\x8a', '\x93', '\x90', '\x83', '\x96',
            '\x9b', '\x9e', '\x99', '\x87', '\x84', '\x9f',
        ]
        for s in spaces:
            x = x.replace(s, " ")
        return x

    def _remove_urls(self, x: pd.Series) -> pd.Series:
        """urlを削除する関数
        https?://
            ?=直前も文字が0回か1回より、http://またはhttps://
        [\w!?/+\-_~=;.,*&@#$%()'[\]]+
            []+: カッコ内の文字のいずれかが一文字以上
            \w: 英単語を構成する文字. [A-Za-z0-9]と同等
            \? \+ \- \. \* \$ \( \) \[ \]: メタキャラクタで使用されているため、エスケープ\をつける
            ! / _ ~ = ; , & @ # % ': メタキャラクタでないため、そのまま入力


        Args:
            x (pd.Series): _description_

        Returns:
            pd.Series: _description_
        """
        x = re.sub("https?://[\w!\?/\+\-_~=;\.,\*&@#\$%\(\)'\[\]]+", "", x)
        return x

    def _clean_puncts(self, x: pd.Series) -> pd.Series:
        puncts = [
            ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
            '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
            '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½',
            'à', '…', '\n', '\xa0', '\t', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬',
            '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000',
            '\u202f', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é',
            '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«', '∙', '）', '↓', '、', '│',
            '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
            '≤', '‡', '√'
        ]
        # puncts = string.punctuation  # stringクラスにもある -> !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
        for punct in puncts:
            x = x.replace(punct, f" {punct} ")
        return x


class CreateSentenceFeatures:
    def __init__(self, config: dict):
        self.primary_key = list(config.primary_key.keys())
        self.sentence_columns = config.sentence_columns

    def execute(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe[self.primary_key + self.sentence_columns].copy()
        for s in self.sentence_columns:
            df = self._create_count_characters(dataframe=df, sentence_column=s)
            df = self._create_word_features(dataframe=df, sentence_column=s)
            df = self._create_punct_features(dataframe=df, sentence_column=s)
            df = self._create_upper_lower_features(dataframe=df, sentence_column=s)
            # df = self._create_paragraph_features(dataframe=df, sentence_column=s)  # 全て0だったため
            df = self._create_sentence_features(dataframe=df, sentence_column=s)
            df = self._create_Syllable_features(dataframe=df, sentence_column=s)
            df = self._create_readability_features(dataframe=df, sentence_column=s)
        df = df.drop(self.sentence_columns, axis=1)
        return df

    def _create_count_characters(self, dataframe: pd.DataFrame, sentence_column: str) -> pd.DataFrame:
        df = dataframe.copy()
        # スペースを除いた文字数
        df[f'{sentence_column}_char_count'] = df[sentence_column].apply(lambda x: len(x.replace(' ', '')))
        return df

    def _create_word_features(self, dataframe: pd.DataFrame, sentence_column: str) -> pd.DataFrame:
        df = dataframe.copy()
        df[f'{sentence_column}_word_count'] = df[sentence_column].apply(lambda x: len(x.split()))
        df[f'{sentence_column}_word_unique_count'] = df[sentence_column].apply(lambda x: len(set(x.split())))
        df[f'{sentence_column}_word_unique_ratio'] = df[f'{sentence_column}_word_unique_count'] / (df[f'{sentence_column}_word_count'] + 1)
        df[f'{sentence_column}_word_ave_length'] = df[sentence_column].apply(lambda x: sum([len(y) for y in x.split()]) / len(x.split()))
        return df

    def _create_punct_features(self, dataframe: pd.DataFrame, sentence_column: str) -> pd.DataFrame:
        df = dataframe.copy()
        punctuations = string.punctuation
        df[f'{sentence_column}_punc_count'] = df[sentence_column].apply(
            lambda x: len([y for y in x.split() if y in punctuations]))
        return df

    def _create_upper_lower_features(self, dataframe: pd.DataFrame, sentence_column: str) -> pd.DataFrame:
        df = dataframe.copy()
        df[f'{sentence_column}_uppercase'] = df[sentence_column].str.findall(r'[A-Z]').str.len()
        df[f'{sentence_column}_lowercase'] = df[sentence_column].str.findall(r'[a-z]').str.len()
        df[f'{sentence_column}_up_low_ratio'] = df[f'{sentence_column}_uppercase'] / (df[f'{sentence_column}_lowercase'] + 1)
        return df

    def _create_paragraph_features(self, dataframe: pd.DataFrame, sentence_column: str) -> pd.DataFrame:
        df = dataframe.copy()
        df[f'{sentence_column}_paragraph_count'] = df[sentence_column].apply(lambda x: x.count('\n'))
        return df

    def _create_sentence_features(self, dataframe: pd.DataFrame, sentence_column: str) -> pd.DataFrame:
        """sentence単位の特徴量を作成する関数

        sent_tokenize
            "This is a pen. Is this a pen?" -> ['This is a pen.', 'Is this a pen?']

        Args:
            dataframe (pd.DataFrame): _description_
            sentence_column (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = dataframe.copy()
        df[f'{sentence_column}_sentence_count'] = df[sentence_column].apply(lambda x: len(sent_tokenize(x)))
        df[f'{sentence_column}_sentence_ave_length'] = df[sentence_column].apply(
            lambda x: sum([len(y) for y in sent_tokenize(x)]) / len(sent_tokenize(x))
        )
        df[f'{sentence_column}_sentence_max_length'] = df[sentence_column].apply(
            lambda x: max([len(y) for y in sent_tokenize(x)])
        )
        df[f'{sentence_column}_sentence_min_length'] = df[sentence_column].apply(
            lambda x: min([len(y) for y in sent_tokenize(x)]))
        df[f'tmp_word_per_sentence'] = df[sentence_column].apply(lambda x: len(word_tokenize(x)))
        df[f'{sentence_column}_word_per_sentence'] = df['tmp_word_per_sentence'] / (df[f'{sentence_column}_sentence_count'] + 1)
        df = df.drop("tmp_word_per_sentence", axis=1)
        return df

    def _create_Syllable_features(self, dataframe: pd.DataFrame, sentence_column: str) -> pd.DataFrame:
        """音節に関する特徴量を作成する関数

        syllable_count:
            - syllableは音節です。音節とは、音として一つのまとまりを表す単位です
            - 'Corgi is beautiful.' -> Cor・gi・is・beau・ti・ful
        Lexicon Count:
            - lexicon_countは、テキストに含まれる単語の数を計算します。
                - オプションのremovepunctで、カウント時に句読点を考慮すべきかどうかを指定します（Trueでカウント前に句読点を削除）
        Sentence Count:
            - sentenceは文です。
            - 'Corgi is beautiful. And, he is also cute.' -> 2

        Args:
            dataframe (pd.DataFrame): _description_
            sentence_column (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = dataframe.copy()
        df[f'{sentence_column}_syllable_count'] = df[sentence_column].apply(lambda x: textstat.syllable_count(x))
        df[f'tmp_sentence_count'] = df[sentence_column].apply(lambda x: textstat.sentence_count(x))
        df[f'tmp_word_count'] = df[sentence_column].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
        df[f'{sentence_column}_syllable_per_sentence'] = df[
            f'{sentence_column}_syllable_count'] / (df["tmp_sentence_count"] + 1)
        df[f'{sentence_column}_syllable_per_word'] = df[
            f'{sentence_column}_syllable_count'] / (df[f'tmp_word_count'] + 1)
        drop_cols = ["tmp_sentence_count", "tmp_word_count"]
        df = df.drop(drop_cols, axis=1)
        return df

    def _create_readability_features(self, dataframe: pd.DataFrame, sentence_column: str) -> pd.DataFrame:
        """読みやすさの特徴量を作成する関数

        参考：https://qiita.com/shoku-pan/items/793a3bcc11a64a3665bf

        Flesch-Kincaid Readability Test（フレッシュ＝キンケード可読性試験）
            - 英語の文章がどれだけ理解しにくいかを示すための読みやすさテストです
            - 大きいほど簡単で、低いほど難しい
        The Flesch-Kincaid Grade Level
            - Flesch Reading-Ease同様、英語の文章がどれだけ理解しにくいかを示すための読みやすさテストです。
            - このスコアは、一読して文章を理解するために必要な正規の教育を受けた年数を推定したものになります。
        Gunning fog index
            - Gunning fog indexも、英語の文章の読みやすさをテストしたものです。
            - 100語程度の文章（1つ以上の段落）を選ぶ。文章は省略しない。など、細かい条件がある。
        Simple Measure of Gobbledygook
            - SMOGは、特にヘルスケア領域で広く利用されているようです。
            - textstatのsmog_indexで有効な結果を得るためには、少なくとも3文が必要です。

        Args:
            dataframe (pd.DataFrame): _description_
            sentence_column (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = dataframe.copy()
        df[f"{sentence_column}_flesch_reading_ease_score"] = df[sentence_column].apply(
            lambda x: textstat.flesch_reading_ease(x))
        # df[f"{sentence_column}_flesch_kincaid_grade_score"] = df[sentence_column].apply(
        #     lambda x: textstat.flesch_kincaid_grade(x))
        df[f"{sentence_column}_gunning_fog_score"] = df[sentence_column].apply(
            lambda x: textstat.gunning_fog(x))
        # 少なくとも3文必要なので
        if sentence_column == "DESCRIPTION_TRANSLATED":
            df[f"{sentence_column}_smog_index_score"] = df[sentence_column].apply(
                lambda x: textstat.smog_index(x))
        return df
