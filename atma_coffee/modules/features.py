import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections.abc import Callable
from common_module.utils.preprocessor import CountEncodingBlock


def make_primary_key(
    input: pd.DataFrame,
    is_train: bool = True
) -> pd.DataFrame:
    """主キーのように扱える列が無いためprimary_key列を作成する関数

    Args:
        df (pd.DataFrame): df_train, df_test
        is_train (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: primary_key列が追加されたdf
    """
    df = input.reset_index()  # index列を作成する
    if is_train:
        df["data_type"] = "train"
    else:
        df["data_type"] = "test"
    df["primary_key"] = df["data_type"] + "_" + df["index"].astype(str)
    return df.drop(["index", "data_type"], axis=1)


def get_world_country_features(
    input: pd.DataFrame,
    primary_key: List[str]
) -> pd.DataFrame:
    """国名から地域名に変換する関数

    Args:
        input (pd.DataFrame): _description_
        primary_key (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    world_region_dict = {
        'Brazil': "South_America",
        'Burundi': "Africa",
        'China': "East_Asia",
        'Colombia': "South_America",
        'Costa Rica': "Central_America",
        'Cote d?Ivoire': "Africa",
        'Ecuador': "South_America",
        'El Salvador': "Central_America",
        'Ethiopia': "Africa",
        'Guatemala': "Central_America",
        'Haiti': "Central_America",
        'Honduras': "Central_America",
        'India': "South_Asia",
        'Indonesia': "SouthEast_Asia",
        'Japan': "East_Asia",
        'Kenya': "Africa",
        'Laos': "SouthEast_Asia",
        'Malawi': "Africa",
        'Mauritius': "Africa",
        'Mexico': "North_America",
        'Myanmar': "SouthEast_Asia",
        'Nicaragua': "Central_America",
        'Panama': "Central_America",
        'Papua New Guinea': "Oseania",
        'Peru': "South_America",
        'Philippines': "SouthEast_Asia",
        'Rwanda': "Africa",
        'Taiwan': "East_Asia",
        'Tanzania, United Republic Of': "Africa",
        'Thailand': "SouthEast_Asia",
        'Uganda': "Africa",
        'United States': "North_America",
        'United States (Hawaii)': "North_America",
        'United States (Puerto Rico)': "North_America",
        'Vietnam': "SouthEast_Asia",
        'Zambia': "Africa"
    }
    df = input[primary_key + ["countryof_origin"]].copy()
    df["world_region"] = df["countryof_origin"].map(world_region_dict)
    output_cols = ["countryof_origin", "world_region"]
    return df[primary_key + output_cols]


def get_altitude_features(
    input: pd.DataFrame,
    primary_key: List[str]
) -> pd.DataFrame:
    """altitude列を取り出す関数.ftからmに変換して単位を揃える。
    A foot was defined as exactly 0.3048 meters in 1959.

    Args:
        input (pd.DataFrame): _description_
        primary_key (List[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    altitude_cols = [
        "altitude_low_meters",
        "altitude_high_meters",
        "altitude_mean_meters"
    ]
    df = input[primary_key + altitude_cols + ["unit_of_measurement"]].copy()

    # ftをmに変換する
    def _convert_ft_to_m(x, col):
        if (x[col] > 0) & (x["unit_of_measurement"] == "ft"):
            return x[col] * 0.3048
        elif (x[col] > 0) & (x["unit_of_measurement"] == "m"):
            return x[col]
        else:
            return x[col]  # 単位がわからない場合はmとする
    
    for col in altitude_cols:
        df[col] = df[col].astype(float)
        df[col + "_m"] = df.apply(lambda x: _convert_ft_to_m(x, col), axis=1)
    
    # output
    output_cols = [
        "altitude_low_meters_m",
        "altitude_high_meters_m",
        "altitude_mean_meters_m"
    ]
    return df[primary_key + output_cols]


def get_num_bags_feature(input: pd.DataFrame) -> pd.DataFrame:
    """bagの数に関する特徴量を作成する関数
    ex) 8 -> 008 -> 0, 0, 8とする

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = input[["primary_key", "numberof_bags"]].copy()
    df["numberof_bags_str"] = [bag.zfill(3) for bag in df["numberof_bags"].astype(str)]
    df["numberof_bags_100"] = df["numberof_bags_str"].apply(lambda x: int(x[0]))
    df["numberof_bags_010"] = df["numberof_bags_str"].apply(lambda x: int(x[1]))
    df["numberof_bags_001"] = df["numberof_bags_str"].apply(lambda x: int(x[2]))
    output_cols = [
        "numberof_bags_100",
        "numberof_bags_010",
        "numberof_bags_001"
    ]
    return df[["primary_key"] + output_cols]


def get_bag_weight_features(input: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    ・()で囲まれたグループが複数あると、各グループで抽出された部分がそれぞれ列となる
    ・[0-9]: 任意の数字, *: ０回以上の繰り返し, \s: 任意の空白文字, [a-zA-Z]: 任意の英字
    ・1lbs = 0.454kg = 454g

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_base = input[["primary_key"]].copy()
    df_bw = input['bag_weight'].str.extract(r'([0-9]*)\s*([a-zA-Z]*)')
    df_bw.columns = ['bag_weight_value', 'bag_weight_unit']
    df_bw["bag_weight_value"] = df_bw["bag_weight_value"].astype(int)

    def _calc_bag_weight_value(x):
        if x["bag_weight_unit"] == "kg":
            return x["bag_weight_value"] * 1000
        elif x["bag_weight_unit"] == "lbs":
            return x["bag_weight_value"] * 0.454 * 1000
        else:
            return x["bag_weight_value"] * 1000  # 単位がわからない場合はkgとしてgに変換する
    
    df_bw["bag_weight_g"] = df_bw.apply(lambda x: _calc_bag_weight_value(x), axis=1)
    df_bw["bag_weight_log_g"] = np.log1p(df_bw["bag_weight_g"])

    # primary_keyとconcat
    df = pd.concat([df_base, df_bw], axis=1)
    output_cols = [
        "bag_weight_unit",
        "bag_weight_g",
        "bag_weight_log_g"
    ]
    return df[["primary_key"] + output_cols]


def get_hervest_year_features(input: pd.DataFrame) -> pd.DataFrame:
    """harvest yearを修正する関数

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    harvest_year_dict = {
        '08/09 crop': 2009,
        '1T/2011': 2011,
        '1t/2011': 2011,
        '2009 - 2010': 2009,
        '2009 / 2010': 2009,
        '2009-2010': 2009,
        '2009/2010': 2009,
        '2010': 2010,
        '2010-2011': 2010,
        '2011': 2011,
        '2011/2012': 2011,
        '2012': 2012,
        '2013': 2013,
        '2013/2014': 2013,
        '2014': 2014,
        '2014/2015': 2014,
        '2015': 2015,
        '2015/2016': 2015,
        '2016': 2016,
        '2016 / 2017': 2016,
        '2016/2017': 2017,
        '2017': 2017,
        '2017 / 2018': 2017,
        '2018': 2018,
        '23 July 2010': 2010,
        '3T/2011': 2011,
        '47/2010': 2010,
        '4T/10': 2010,
        '4T/2010': 2010,
        '4T72010': 2010,
        '4t/2010': 2010,
        '4t/2011': 2011,
        'Abril - Julio': np.nan,
        'Abril - Julio /2011': 2011,
        'August to December': np.nan,
        'December 2009-March 2010': 2009,
        'Fall 2009': 2009,
        'January 2011': 2011,
        'January Through April': np.nan,
        'March 2010': 2010,
        'May-August': np.nan,
        'Mayo a Julio': np.nan,
        'Sept 2009 - April 2010': 2009,
        'Spring 2011 in Colombia.': 2011,
        'TEST': np.nan,
        'mmm': np.nan
    }
    df = input[["primary_key", "harvest_year"]].copy()
    df["harvest_year"] = df["harvest_year"].map(harvest_year_dict)
    return df


def get_grading_date_features(input: pd.DataFrame) -> pd.DataFrame:
    """grading_dateの特徴量を作成する関数
    ※grading_date列にヌルはない

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = input[["primary_key", "grading_date"]].copy()
    df["grading_date"] = pd.to_datetime(df["grading_date"])
    df["grading_date_julian"] = df["grading_date"].map(pd.Timestamp.to_julian_date)
    df["grading_date_year"] = df["grading_date"].dt.year
    df["grading_date_month"] = df["grading_date"].dt.month
    df["grading_date_day"] = df["grading_date"].dt.day
    # df["grading_date_yyyymmdd"] = df["grading_date_year"].astype(str) + df["grading_date_month"].astype(str) + df["grading_date_day"].astype(str)
    # df["grading_date_yyyymmdd"] = df["grading_date_yyyymmdd"].astype(int)
    output_cols = [
        "grading_date_julian",
        "grading_date_year",
        "grading_date_month",
        "grading_date_day"
    ]
    return df[["primary_key"] + output_cols]


def get_expiration_features(input: pd.DataFrame) -> pd.DataFrame:
    """expirationの特徴量を作成する関数
    ※expiration列にヌルはない

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = input[["primary_key", "expiration"]].copy()
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["expiration_julian"] = df["expiration"].map(pd.Timestamp.to_julian_date)
    df["expiration_year"] = df["expiration"].dt.year
    df["expiration_month"] = df["expiration"].dt.month
    df["expiration_day"] = df["expiration"].dt.day
    # df["expiration_yyyymmdd"] = df["expiration_year"].astype(str) + df["expiration_month"].astype(str) + df["expiration_day"].astype(str)
    # df["expiration_yyyymmdd"] = df["expiration_yyyymmdd"].astype(int)
    output_cols = [
        "expiration_julian",
        "expiration_year",
        "expiration_month",
        "expiration_day"
    ]
    return df[["primary_key"] + output_cols]


def get_moisture_features(input: pd.DataFrame) -> pd.DataFrame:
    """moisture列の特徴量を作成する関数
    ※moisture列にヌルはない

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = input[["primary_key", "moisture"]].copy()
    df["moisture_010"] = [int(str(t)[2]) for t in df["moisture"]]
    output_cols = ["moisture_010"]
    return df[["primary_key"] + output_cols]


def get_defects_features(input: pd.DataFrame) -> pd.DataFrame:
    """欠点に関する特徴量を作成する関数
    ※one: 主要欠点。多いほど評価が悪い。two: マイナー欠点。多いほど評価が悪い。
    ※両列にヌルはない
    https://www.coffeestrategies.com/wp-content/uploads/2020/08/Green-Coffee-Defect-Handbook.pdf
    category one defect: 

    Args:
        input (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = input[["primary_key", "category_one_defects", "category_two_defects"]].copy()
    # category one defectsの方が致命的に味へ影響するため
    df["category_one_and_two_defects"] = (df["category_one_defects"] * 5) + (df["category_two_defects"] * 1)
    output_cols = ["category_one_and_two_defects"]
    return df[["primary_key"] + output_cols]


def get_cross_num_features(input: pd.DataFrame) -> pd.DataFrame:
    df = input[[
        "primary_key",
        "harvest_year",  # get_hervest_year_features
        "grading_date_year",  # get_grading_date_features
        "grading_date_julian",
        "expiration_year",  # get_expiration_features
        "expiration_julian",
        "numberof_bags",  #  get_num_bags_feature
        "bag_weight_g",  # get_bag_weight_features
        "bag_weight_log_g",
    ]].copy()

    # calc
    df["diff_hervest_grading_year"] = df["grading_date_year"] - df["harvest_year"]
    df["diff_hervest_expiration_year"] = df["expiration_year"] - df["harvest_year"]
    df["diff_grading_expiration_year"] = df["expiration_year"] - df["grading_date_year"]
    # df["diff_grading_expiration_julian"] = df["expiration_julian"] - df["grading_date_julian"]  # 全て365のため削除
    df["total_weight_g"] = df["numberof_bags"] * df["bag_weight_g"]
    df["total_weight_log_g"] = df["numberof_bags"] * df["bag_weight_log_g"]
    output_cols = [
        "diff_hervest_grading_year",
        "diff_hervest_expiration_year",
        "diff_grading_expiration_year",
        # "diff_grading_expiration_julian",
        "total_weight_g",
        "total_weight_log_g"
    ]
    return df[["primary_key"] + output_cols]


def merge_features(
    input: pd.DataFrame,
    is_train: bool,
    config: Dict[str, List[str]],
    encoder_ceb: Callable = None
):
    # 特徴量を作成する
    df_country = get_world_country_features(input, config.primary_key)
    df_altitude = get_altitude_features(input, config.primary_key)
    df_bags = get_num_bags_feature(input)
    df_bagWeight = get_bag_weight_features(input)
    df_harvest = get_hervest_year_features(input)
    df_granding = get_grading_date_features(input)
    df_exp = get_expiration_features(input)
    df_moisture = get_moisture_features(input)
    df_defects = get_defects_features(input)

    # 特徴量をマージする
    if is_train:
        df_base = input[config.primary_key + config.origin_cols + config.pred_cols].copy()
    else:
        df_base = input[config.primary_key + config.origin_cols].copy()
    df_merged = pd.merge(df_base, df_country, how="left", on=config.primary_key)
    df_merged = pd.merge(df_merged, df_altitude, how="left", on=config.primary_key)
    df_merged = pd.merge(df_merged, df_bags, how="left", on=config.primary_key)
    df_merged = pd.merge(df_merged, df_bagWeight, how="left", on=config.primary_key)
    df_merged = pd.merge(df_merged, df_harvest, how="left", on=config.primary_key)
    df_merged = pd.merge(df_merged, df_granding, how="left", on=config.primary_key)
    df_merged = pd.merge(df_merged, df_exp, how="left", on=config.primary_key)
    df_merged = pd.merge(df_merged, df_moisture, how="left", on=config.primary_key)
    df_merged = pd.merge(df_merged, df_defects, how="left", on=config.primary_key)

    # 作成した特徴量の組み合わせで作成する特徴量
    df_cross_features = get_cross_num_features(input=df_merged)
    df_merged = pd.merge(df_merged, df_cross_features, how="left", on=config.primary_key)

    ins_ceb = CountEncodingBlock()
    if not encoder_ceb:
        encoder_ceb, df_ceb = ins_ceb.fit_transform(df_merged, target_cols=config.count_encoding_target_cols)
        df_merged = pd.concat([df_merged, df_ceb], axis=1)
        return df_merged, encoder_ceb
    else:
        df_ceb = ins_ceb.transform(df_merged, target_cols=config.count_encoding_target_cols, encoder=encoder_ceb)
        df_merged = pd.concat([df_merged, df_ceb], axis=1)
        return df_merged
