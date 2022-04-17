from torch.utils.data import Dataset
from torchvision.io import read_image

import os
import pandas as pd
import numpy as np
import zipfile


class MakeImgDataset:
    def __init__(self, config: dict):
        # path
        self.train_csv_path = config.train_csv_path
        self.test_csv_path = config.train_csv_path
        self.train_zip_path = config.train_zip_path
        self.test_zip_path = config.test_zip_path
        self.input_path = config.input_path
        # csvのopen
        self.train_csv = pd.read_csv(self.train_csv_path, encoding="utf-8-sig")
        self.test_csv = pd.readcsv(self.test_csv_path, encoding="utf-8-sig")

    @staticmethod
    def _get_image_id_list(zip_path: str):
        # image pathを作成
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            image_id_list = []
            # zipファイル内の各ファイルについてループ
            for idx, file_info in enumerate(zip_file.infolist()):
                # 1つ目はzipファイル名のみのためパス
                if idx > 0:
                    file_name = file_info.filename  # <ZipInfo filename='train_images/2833245.jpg' ...
                    image_id = int(file_name.split("/")[1].split(".")[0])
                    image_id_list.append(image_id)
        return image_id_list

    def run(self):
        # dataframe
        df_train = self.train_csv[["LOAN_ID", "IMAGE_ID", "LOAN_AMOUNT"]].copy()
        df_test = self.test_csv[["LOAN_ID", "IMAGE_ID"]].copy()
        # 画像IDリスト
        train_image_id_list = self._get_image_id_list(zip_path=self.train_zip_path)
        test_image_id_list = self._get_image_id_list(zip_path=self.test_zip_path)
        # 画像のあるデータに絞る
        df_train = df_train[df_train["IMAGE_ID"].isin(train_image_id_list)]
        df_test = df_test[df_test["IMAGE_ID"].isin(test_image_id_list)]
        return df_train, df_test
