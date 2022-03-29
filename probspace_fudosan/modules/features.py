import pandas as pd
import numpy as np
import unicodedata
from typing import List, Tuple, Dict


class FeatureEngineering:
    @staticmethod
    def fix_torihiki_jiten(input_df: pd.DataFrame) -> pd.DataFrame:
        """「取引時点」を修正する関数
        2017年第１四半期 -> 2017.0
        2017年第４四半期 -> 2017.75

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        # 文字列を変換しfloatの年代に変えるための準備をする
        df["取引時点"] = df["取引時点"].str.replace("年第", ".")
        df["取引時点"] = df["取引時点"].str.replace("四半期", "")
        df["取引時点"] = df["取引時点"].str.replace("１", "00")
        df["取引時点"] = df["取引時点"].str.replace("２", "25")
        df["取引時点"] = df["取引時点"].str.replace("３", "50")
        df["取引時点"] = df["取引時点"].str.replace("４", "75")
        # floatの年代に変える
        df["取引時点"] = pd.to_numeric(df["取引時点"], errors="raise")
        print("finish fix_torihiki_jiten")
        return df

    @staticmethod
    def fix_menseki(input_df: pd.DataFrame) -> pd.DataFrame:
        """「面積（㎡）」列を修正する関数
        1. 2000㎡以上, 5000㎡以上という文字列があるので修正する
        2. 林地、農地は面積が広い割に価格が小さいため。

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        # 文字列の修正
        df["面積（㎡）"] = df["面積（㎡）"].str.replace("㎡以上", "")
        df["面積（㎡）"] = pd.to_numeric(df["面積（㎡）"], errors="raise")
        # 価格の修正
        df.loc[df["種類"].isin(["林地", "農地"]), "面積（㎡）"] *= 0.1
        print("finish fix_menseki")
        return df

    @staticmethod
    def fix_moyori_eki_kyori(input_df: pd.DataFrame) -> pd.DataFrame:
        """「最寄駅：距離（分）」列を修正する関数
        30分?60分, 1H30?2Hなどの文字列が含まれているため修正する

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        # 文字列を変換しnumericに変えるための準備をする
        df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].replace("30分?60分", "45")
        df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].replace("1H30?2H", "105")
        df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].replace("1H?1H30", "75")
        df["最寄駅：距離（分）"] = df["最寄駅：距離（分）"].replace("2H?", "120")
        # numericにする
        df["最寄駅：距離（分）"] = pd.to_numeric(df["最寄駅：距離（分）"], errors="raise")
        print("finish fix_moyori_eki_kyori")
        return df

    @staticmethod
    def create_madori_features(input_df: pd.DataFrame) -> pd.DataFrame:
        """_summary_

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        df["間取り"] = df["間取り"].fillna("unknown")  # nanがfloatになり変換できないため
        df["間取り"] = df["間取り"].map(lambda x: unicodedata.normalize("NFKC", x))  # 表記揺れ修正
        # 1DK -> 1, DKの特徴量をそれぞれ作る
        df["create_間取り数"] = df["間取り"].map(
            lambda x: int(x[0]) if x[0] in [str(s) for s in range(1, 8)] else np.nan)
        df["create_間取りタイプ"] = df["間取り"].map(
            lambda x: x[1:] if x not in ["オープンフロア", "スタジオ", "メゾネット", "unknown"] else "unknown")
        # 間取りごとに数値を決め、部屋数を算出
        madori_dict = {
            'D': 0,
            'DK': 1,
            'DK+S': 2,
            'K': 0,
            'K+S': 1,
            'L': 1.5,
            'L+S': 2.5,
            'LD': 1.5,
            'LD+S': 2.5,
            'LDK': 1.5,
            'LDK+K': 2.5,
            'LDK+S': 2.5,
            'LK': 1.5,
            'LK+S': 2.5,
            'R': 0,
            'R+S': 1,
            'unknown': 2
        }
        df["tmp_間取りタイプ数値換算"] = df["create_間取りタイプ"].map(madori_dict)
        df["create_部屋数"] = df["create_間取り数"] + df["tmp_間取りタイプ数値換算"]
        df = df.drop("tmp_間取りタイプ数値換算", axis=1)
        print("finish create_madori_features")
        return df

    @staticmethod
    def fix_maguchi(input_df: pd.DataFrame) -> pd.DataFrame:
        """「間口」列の修正

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        df["間口"] = df["間口"].str.replace("m以上", "")
        df["間口"] = pd.to_numeric(df["間口"], errors="raise")
        print("finish fix_maguchi")
        return df

    @staticmethod
    def fix_nobeyuka_menseki(input_df: pd.DataFrame) -> pd.DataFrame:
        """「延床面積（㎡）」列の修正

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        df["延床面積（㎡）"] = df["延床面積（㎡）"].str.replace("㎡以上", "")
        df["延床面積（㎡）"] = df["延床面積（㎡）"].replace("10m^2未満", "5")
        df["延床面積（㎡）"] = pd.to_numeric(df["延床面積（㎡）"], errors="raise")
        print("finish fix_nobeyuka_menseki")
        return df

    @staticmethod
    def fix_create_kenchiku_nensu(input_df: pd.DataFrame) -> pd.DataFrame:
        """「建築年」列の修正と新しい特徴量作成

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        df["建築年"] = df["建築年"].str.replace("戦前", "昭和20年")
        df["create_年号"] = df["建築年"].str[:2]
        df["create_和暦年数"] = pd.to_numeric(df["建築年"].str[2:].str.strip("年"), errors="raise")
        df.loc[df["create_年号"] == "昭和", "建築年（西暦）"] = df["create_和暦年数"] + 1925
        df.loc[df["create_年号"] == "平成", "建築年（西暦）"] = df["create_和暦年数"] + 1988
        print("finish fix_create_kenchiku_nensu")
        return df

    @staticmethod
    def fix_moyori_eki_meisho(input_df: pd.DataFrame) -> pd.DataFrame:
        """「最寄駅：名称」列の修正する関数

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        df["最寄駅：名称"] = df["最寄駅：名称"].str.replace("(東京)", "")
        df["最寄駅：名称"] = df["最寄駅：名称"].str.replace("(神奈川)", "")
        df["最寄駅：名称"] = df["最寄駅：名称"].str.replace("ケ", "ヶ")
        df["最寄駅：名称"] = df["最寄駅：名称"].str.replace("(メトロ)", "")
        df["最寄駅：名称"] = df["最寄駅：名称"].str.replace("(都電)", "")
        df["最寄駅：名称"] = df["最寄駅：名称"].str.replace("(つくばＥＸＰ)", "")
        df["最寄駅：名称"] = df["最寄駅：名称"].str.replace("(千葉)", "")
        df["最寄駅：名称"] = df["最寄駅：名称"].str.replace("(東京メトロ)", "")
        df["最寄駅：名称"] = df["最寄駅：名称"].str.replace("(東武・都営・メトロ)", "")
        df["最寄駅：名称"] = df["最寄駅：名称"].str.strip("()")
        print("finish fix_moyori_eki_meisho")
        return df

    @staticmethod
    def create_shichoson(input_df: pd.DataFrame) -> pd.DataFrame:
        """「市区町村名」列から特徴量を作る関数

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        df[["市区町村名", "地区名"]] = df[["市区町村名", "地区名"]].fillna("unknown")
        df["create_地区詳細"] = df.apply(lambda x: x["市区町村名"] + "_" + x["地区名"], axis=1)
        df["create_is_市"] = np.where(df["市区町村名"].str.contains("市"), 1, 0)
        df["create_is_区"] = np.where(df["市区町村名"].str.contains("区"), 1, 0)
        df["create_is_町"] = np.where(df["市区町村名"].str.contains("町"), 1, 0)
        df["create_is_村"] = np.where(df["市区町村名"].str.contains("村"), 1, 0)
        print("finish create_shichoson")
        return df

    @staticmethod
    def fix_create_tatemono_kouzo(input_df: pd.DataFrame) -> pd.DataFrame:
        """「建物の構造」列の新しい特徴量を作成する、修正する関数
        SRC、RCなどがある。SRCとRCは一部文字が重複しているため検索がやや複雑になった

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        df["建物の構造"] = df["建物の構造"].fillna("unknown")
        df["建物の構造"] = df["建物の構造"].map(lambda x: unicodedata.normalize("NFKC", x))  # 表記揺れ修正
        df["create_is_SRC"] = np.where(df["建物の構造"].str.contains("SRC"), 1, 0)
        df["create_is_RC"] = np.where(df["建物の構造"].str.contains("RC") & ~df["建物の構造"].str.startswith("S"), 1, 0)
        df["create_is_鉄骨造"] = np.where(df["建物の構造"].str.contains("鉄骨造"), 1, 0)
        df["create_is_木造"] = np.where(df["建物の構造"].str.contains("木造"), 1, 0)
        df["create_is_軽量鉄骨造"] = np.where(df["建物の構造"].str.contains("軽量鉄骨造"), 1, 0)
        df["create_is_ブロック造"] = np.where(df["建物の構造"].str.contains("ブロック造"), 1, 0)
        print("finish fix_create_tatemono_kouzo")
        return df

    @staticmethod
    def fix_create_yoto(input_df: pd.DataFrame) -> pd.DataFrame:
        """「用途」列の修正及び新しい特徴量作成を行う関数

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        df["用途"] = df["用途"].fillna("unknown")
        df["用途"] = df["用途"].map(lambda x: unicodedata.normalize("NFKC", x))  # 表記揺れ修正
        df["create_is_住宅"] = np.where(df["用途"].str.contains("住宅") & ~df["用途"].str.startswith("住"), 1, 0)
        df["create_is_共同住宅"] = np.where(df["用途"].str.contains("共同住宅"), 1, 0)
        df["create_is_事務所"] = np.where(df["用途"].str.contains("事務所"), 1, 0)
        df["create_is_店舗"] = np.where(df["用途"].str.contains("店舗"), 1, 0)
        df["create_is_その他"] = np.where(df["用途"].str.contains("その他"), 1, 0)
        df["create_is_倉庫"] = np.where(df["用途"].str.contains("倉庫"), 1, 0)
        df["create_is_駐車場"] = np.where(df["用途"].str.contains("駐車場"), 1, 0)
        df["create_is_作業場"] = np.where(df["用途"].str.contains("作業場"), 1, 0)
        df["create_is_工場"] = np.where(df["用途"].str.contains("工場"), 1, 0)
        print("finish fix_create_yoto")
        return df

    @staticmethod
    def fix_create_torihiki_jizyo(input_df: pd.DataFrame) -> pd.DataFrame:
        """「取引の事情等」列の修正及び新しい特徴量の作成を行う関数

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        df["取引の事情等"] = df["取引の事情等"].fillna("unknown")
        df["取引の事情等"] = df["取引の事情等"].map(lambda x: unicodedata.normalize("NFKC", x))  # 表記揺れ修正
        df["create_is_私道を含む取引"] = np.where(df["取引の事情等"].str.contains("私道を含む取引"), 1, 0)
        df["create_is_隣地の購入"] = np.where(df["取引の事情等"].str.contains("隣地の購入"), 1, 0)
        df["create_is_関係者間取引"] = np.where(df["取引の事情等"].str.contains("関係者間取引"), 1, 0)
        df["create_is_調停・競売等"] = np.where(df["取引の事情等"].str.contains("調停・競売等"), 1, 0)
        df["create_is_その他事情有り"] = np.where(df["取引の事情等"].str.contains("その他事情有り"), 1, 0)
        df["create_is_瑕疵有りの可能性"] = np.where(df["取引の事情等"].str.contains("瑕疵有りの可能性"), 1, 0)
        df["create_is_古屋付き・取壊し前提"] = np.where(df["取引の事情等"].str.contains("古屋付き・取壊し前提"), 1, 0)
        df["create_is_他の権利・負担付き"] = np.where(df["取引の事情等"].str.contains("他の権利・負担付き"), 1, 0)
        print("finish fix_create_torihiki_jizyo")
        return df

    @staticmethod
    def create_combination_features(input_df: pd.DataFrame) -> pd.DataFrame:
        """様々な列の特徴量を掛け合わせて特徴量を作成する関数

        Args:
            input_df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df = input_df.copy()
        # 階数を求める
        df["create_階数"] = df["延床面積（㎡）"] / df["面積（㎡）"]
        # 全体面積に対する間口の広さ => 狭い間口だと土地を買っても法律の関係で家を立てられない場合があり、結果売れにくいため
        df["create_間口面積割合"] = df["間口"] / df["面積（㎡）"]
        # 各部屋の平均の広さ
        df["create_1部屋面積"] = df["延床面積（㎡）"] / df["create_部屋数"]
        print("finish create_combination_features")
        return df
