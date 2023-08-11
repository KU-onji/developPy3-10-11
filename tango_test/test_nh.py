import os
import re
import random
import pandas as pd

TANGO_LIST = "tango_test/list2"


def main():
    df = get_wordcsv()
    print("学年を入力してください。")
    g = input_num([1, 2, 3], "学年")
    print("Unitを入力してください。入力例 (1と3～5): 1, 3-5")
    u = input_unit(df, g)
    df = df_filter(df, g, u)
    test = select_test(df)
    print("テスト形式を選んでください。")
    print("1: 読みを答えるテスト (英 → 日)")
    print("2: スペルを答えるテスト (日 → 英)")
    sel = input_num([1, 2], "テスト形式")
    if sel == 0:
        print_read_test(df, test)
    else:
        print_write_test(test)


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
    return [
        pd.read_csv(os.path.join(folderpath, file), encoding="utf-8")
        for file in files
    ]


def get_grade_unit(row):
    m = re.match(r"([0-9]*)U([0-9]*)", row)
    return int(m.group(1)), int(m.group(2))


def split_series(row):
    return pd.Series(row)


def clean_df(dfs):
    for df in dfs:
        df[["Grade", "Unit"]] = (
            df["単元"].apply(get_grade_unit).apply(split_series)
        )
        df.drop("単元", axis=1, inplace=True)
        df.set_index(["Grade", "Unit"], inplace=True)


def save_dfs(dfs, folderpath):
    curr = os.getcwd()
    folder = os.path.join(curr, folderpath)
    for i, df in enumerate(dfs):
        filename = f"wordlist_NH_{i}.csv"
        df.to_csv(os.path.join(folder, filename))


def get_wordcsv():
    curr = os.getcwd()
    file = os.path.join(curr, "tango_test/list2/wordlist_NH_0.csv")
    return pd.read_csv(file)


def input_unit(df, g):
    start = df.groupby("Grade")["Unit"].min()[g]
    end = df.groupby("Grade")["Unit"].max()[g]
    u = input("Unit: ")
    ls = re.split(r"\s*,\s*", u)
    res = []
    try:
        for item in ls:
            if re.fullmatch(r"\d+-\d+", item):
                s, e = map(int, item.split("-"))
                res += list(range(s, e + 1))
            elif re.fullmatch(r".*-.*", item):
                print(f"エラー: {item}")
                print("範囲指定は3-5のように指定してください")
                raise ValueError
            elif re.fullmatch(r"\D+", item):
                print(f"エラー: {item}")
                print("半角数字で入力してください。")
                raise ValueError
            else:
                res += [int(item)]
        if (min(res) < start) | (end < max(res)):
            print(f"Unitの範囲は{start}から{end}までです。")
            raise ValueError
        else:
            return res
    except ValueError:
        print("もう一度Unit番号を入力してください。")
        return input_unit(df, g)


def df_filter(df, g, u):
    return df[(df["Grade"] == g) & df["Unit"].isin(u)]


def select_test(df):
    if len(df) < 20:
        return df.sample(n=len(df))
    else:
        return df.sample(n=20)


def input_num(ls, title):
    try:
        num = int(input(f"{title}: "))
        if num in ls:
            return num
        else:
            print(f"{title}は{min(ls)}～{max(ls)}の範囲で入力してください。")
            return input_num(ls, title)
    except ValueError:
        print(f"{title}は{min(ls)}～{max(ls)}の半角数字で入力してください。")
        return input_num(ls, title)


def print_read_test(df, test):
    score = 0
    for i, idx in enumerate(test.index):
        word = test["英語"][idx]
        ans = test["日本語"][idx]
        opt = list(df["日本語"].sample(n=3))
        while set(ans) <= set(opt):
            opt = list(df["日本語"].sample(n=3))
        opt.append(ans)
        random.shuffle(opt)
        print(f"{f'{i+1}問目: ':>7} {word}")
        for j in range(4):
            print(f"{f'[{j+1}]: ':>7}", opt[j])
        res = input_num([1, 2, 3, 4], "選択肢") - 1
        if opt[res] == ans:
            print("正解！")
            score += 1
        else:
            print("不正解……")
            print(f"正解は「{ans}」でした。")
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


def list_equal(l1, l2):
    if len(l1) != len(l2):
        return False
    else:
        return all(x == y for x, y in zip(l1, l2))


def print_write_test(test):
    score = 0
    char = re.compile(r"[a-zA-Z,\.\?]+")
    for i, idx in enumerate(test.index):
        meaning = test["日本語"][idx]
        word = test["英語"][idx]
        ans = re.findall(char, word)
        print(f"{f'{i+1}問目: ':>7} {meaning}")
        res = input(f"{'英語: ':>7}")
        tmp = re.findall(char, res)
        if list_equal(tmp, ans):
            print("正解！")
            score += 1
        else:
            print("不正解……")
            print(f"正解は「{word}」でした。")
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


# prepro()
if __name__ == "__main__":
    main()
