import os
import re

import pandas as pd

TANGO_LIST = "tango_test/list2"


def main():
    pass


def prepro():
    files = get_files(TANGO_LIST)
    dfs = get_dfs(TANGO_LIST, files)
    clean_df(dfs)
    save_dfs(dfs, TANGO_LIST)


def get_files(folderpath):
    curr = os.getcwd()
    folder = os.path.join(curr, folderpath)
    return [file for file in os.listdir(folder) if file.endswith(".csv")]


def get_dfs(folderpath, files):
    return [pd.read_csv(os.path.join(folderpath, file), encoding="utf-8") for file in files]


def get_grade_unit(row):
    m = re.match(r"([0-9]*)U([0-9]*)", row)
    return m.group(1), m.group(2)


def split_series(row):
    return pd.Series(row)


def clean_df(dfs):
    for df in dfs:
        df[["Grade", "Unit"]] = df["単元"].apply(get_grade_unit).apply(split_series)
        df.drop("単元", axis=1, inplace=True)
        df.set_index(["Grade", "Unit"], inplace=True)


def save_dfs(dfs, folderpath):
    curr = os.getcwd()
    folder = os.path.join(curr, folderpath)
    for i, df in enumerate(dfs):
        filename = f"wordlist_NH_{i}.csv"
        df.to_csv(os.path.join(folder, filename))


# prepro()
