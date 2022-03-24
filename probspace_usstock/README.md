## EDA
### company list
- Symbolが重複している場合がある
    - Listが異なっている
- nullがある列がある
    - IPOyear
    - Sector
    - Industry: 粒度がかなり細かいので特徴量で使うと過学習しそう

## 目的変数の設定
- 収益率(=log(当日) - log(前日))が良さそう
    - 計算方法(1->2)
        - 1. 対数化（np.log1p）※戻すのはnp.expm1
        - 2. 前日との差分（diff）
    - 理由
        - トレンド排除
        - 定常化
        - 桁数の抑制

## 特徴量
- category
    - company関連
        - Sector
        - List
    - ymd関連(yearは例えば2017年まで学習して2018年を予測すると精度が悪化する可能性があるため使わない)
        - month
        - day
        - week_of_month
        - week_of_year
- float64
    - IOPyear
    - 1週間前
    - 2週間前
    - Symbolの4, 8, 12週間の平均値, 中央値, 標準偏差
    - 同Sectorの1週間前の平均値, 中央値, 標準偏差
    - 同Listの1週間前の平均値, 中央値, 標準偏差

## train, validation, test期間
※リークするためクロスバリデーションはできない
- train
    - 2011/11/13 ~ 2018/12/30
- validation
    - 2019/1/6 ~ 2019/11/10
- test
    - 2019/11/17

## memo
- データ理解
    - 株価にnullはあるか？
    - company listのprofile作成
- 上位解法の読み込み
    - 2, 3位を参考に
- model
    - LightGBM
    - transformer
- その他やっておきたいこと
    - loggerの作成