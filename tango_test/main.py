import pandas as pd
import os
import random
import re
import json


def main():
    files = get_files(TANGO_LIST)
    keys = get_dfkeys(TANGO_LIST)
    dfs = get_dfs(TANGO_LIST, files)

    for df in dfs:
        df.drop("#", axis=1, inplace=True)

    vocab_list = make_vocab_list(dfs, keys)

    random.shuffle(vocab_list)
    test50 = [vocab_list[i : i + 50] for i in range(0, len(vocab_list), 50)]
    # save_tests(test50, TANGO_LIST)
    test20 = list(range(len(test50)))
    opt20 = list(range(len(test50)))
    for i in range(len(test50)):
        test20[i], opt20[i] = make_test(test50[i])

    print_test(test20, opt20, N)


def get_files(folderpath):
    curr = os.getcwd()
    folder = os.path.join(curr, folderpath)
    return [file for file in os.listdir(folder) if file.endswith(".csv")]


def get_dfkeys(folderpath):
    curr = os.getcwd()
    folder = os.path.join(curr, folderpath)
    k = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            m = re.match(r"(.*)リスト.csv", file)
            k.append(m.group(1))
    return k


def get_dfs(folderpath, files):
    return [
        pd.read_csv(os.path.join(folderpath, file), encoding="utf-8")
        for file in files
    ]


def make_vocab_list(dfs, keys):
    d = {}
    for i, df in enumerate(dfs):
        words = list(df[keys[i]])
        meanings = list(df["意味"])
        pairs = list(zip(words, meanings))
        d[keys[i]] = pairs
    res = []
    for key in d.keys():
        res = res + d[key]
    return res


def save_tests(test, folderpath):
    curr = os.getcwd()
    folder = os.path.join(curr, folderpath)
    for i in range(len(test)):
        with open(
            folder + f"/tests/test50_{i}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(test[i], f, ensure_ascii=False, indent=4)


def make_test(ls):
    LENGTH = 20
    random.shuffle(ls)
    test = ls[0:LENGTH]
    ans = [test[i][1] for i in range(LENGTH)]
    opt = []
    for i in range(LENGTH):
        cnd = random.sample(ls, 3)
        cnd = [cnd[j][1] for j in range(3)]
        while set([ans[i]]) <= set(cnd):
            cnd = random.sample(ls, 3)
            cnd = [cnd[j][1] for j in range(3)]
        opt.append(list(cnd) + [ans[i]])
        random.shuffle(opt[i])
    return test, opt


def print_test(test, opt, n):
    indent = "        "
    score = 0
    for i in range(len(test[n])):
        print(f"{f'{i+1}問目: ':>7} {test[n][i][0]}")
        for j in range(4):
            s = opt[n][i][j]
            print(f"{f'[{j+1}]: ':>7}", s.replace("\n", "\n" + indent))
        while True:
            try:
                res = int(input()) - 1
                if res < 0:
                    raise (IndexError)
                if test[n][i][1] == opt[n][i][res]:
                    print("正解！")
                    score += 1
                else:
                    print("不正解……")
                    print(f"正解は「{test[n][i][1]}」でした。")
                break
            except IndexError:
                print("1から4の範囲で入力してください。")
            except ValueError:
                print("1から4の数字を入力してください。")
    print(f"{score}点でした。", end="")
    if score <= 5:
        print("もうちょっと頑張りましょう。")
    elif score <= 10:
        print("ぼちぼちできていますね。")
    elif score <= 15:
        print("結構良い成績！")
    elif score < 19:
        print("よくできています！")
    else:
        print("素晴らしい！！！よく勉強しました。")


SEED = 777
random.seed(SEED)
TANGO_LIST = "tango_test/list"
KEYS = [
    "代名詞",
    "冠詞",
    "前置詞",
    "副詞",
    "助動詞",
    "動詞",
    "名詞",
    "形容詞",
    "接続詞",
    "間投詞",
]
N = 0

if __name__ == "__main__":
    main()
