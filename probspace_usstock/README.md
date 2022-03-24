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
    - 同Listの1週間前の平均値, 中央値, 標準偏差

## train, validation, test期間
※リークするためクロスバリデーションはできない
- train
    - 2011/11/13 ~ 2018/12/30
- validation
    - 2019/1/6 ~ 2019/11/10
- test
    - 2019/11/17

## 精度改善案
- 改善前のスコア(Baseline)
    - 2019/11/17:
    - Public: 0.05133
    - Private: 0.04963

- 1. validation期間をもっと短くする
    - 詳細
        - train_period: ['2011-11-13', '2019-08-31']
        - valid_period: ['2019-09-01', '2019-11-10']
    - 結果
        - 2019/11/17: 0.05419
        - Public: 0.04929
        - Private: 0.04712
    - 結論
        - 採用する

- 2. yearを追加する
    - 詳細
        - yearを追加することで年毎のトレンドを取り込む
    - 結果
        - 2019/11/17: 0.05304
        - Public: 0.04926
        - Private: 0.04465
    - 結論
        - 採用する

- 3. ymd関連の特徴量をcategory -> intにする
    - 詳細
        - category -> intにする
    - 結果
        - 2019/11/17: 0.0507
        - Public: 0.04976
        - Private: 0.04528
    - 結論
        - 採用しない

- 4. stock_priceの値が極端に小さいまたは大きいものがうまく予測できていないので、log変換する
    - 詳細
        - np.log1p() -> <- np.expm1()
    - 結果
        - 2019/11/17: 0.0181
        - Public: 0.04766
        - Private: 0.04524
    - 結論
        - 採用する

- 5. Sectorを加える
    - 詳細
        - Sectorごとの特徴を組み込む
    - 結果
        - 2019/11/17: 0.0177
        - Public: 0.04766
        - Private: 0.04524
    - 結論
        - 

- 6. stock_priceの値が極端に小さいまたは大きいものがうまく予測できていないので、フラグを作る
    - 詳細
        - stock_priceをlog1p変換後の分布
            - 10%: 1.3
            - 90%: 4.6
        - 1.6 <: is_10_percentile_stock_price
        - 4.2 >: is_90_percentile_stock_price
    - 結果
        - 2019/11/17:　0.0177
        - Public:
        - Private:
    - 結論
        - 

以下やっていないこと
- 6. Listの特徴量を加える
    - 詳細
        - 同List市場がどのようなトレンドか調べる
        - valid_period: ['2019-09-01', '2019-11-10']
    - 結果
        - 2019/11/17:
        - Public: 
        - Private: 
    - 結論
        - 

- 7. 騰落率の特徴量を加える
    - 詳細
        - train_period: ['2011-11-13', '2019-08-31']
        - valid_period: ['2019-09-01', '2019-11-10']
    - 結果
        - 2019/11/17:
        - Public: 
        - Private: 
    - 結論
        - 

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