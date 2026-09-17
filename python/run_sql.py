# -*- coding: utf-8 -*-
"""
SQL 执行器：按 `-- @query: 名称` 分块执行 sql/ 目录下的分析脚本并打印结果。

Usage:
    python run_sql.py                  # 跑全部
    python run_sql.py 01_funnel        # 跑文件名包含 01_funnel 的脚本
    python run_sql.py --save           # 同时把结果落到 output/ 目录（csv）
"""
import os
import re
import sys
import glob
import duckdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_DIR = os.path.join(BASE_DIR, "sql")
OUT_DIR = os.path.join(BASE_DIR, "output")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.unicode.east_asian_width", True)


def split_queries(text):
    """按 `-- @query: name` 切分成 (name, sql) 列表"""
    parts = re.split(r"^--\s*@query:\s*(.+)$", text, flags=re.MULTILINE)
    out = []
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        body = parts[i + 1].strip()
        if body:
            out.append((name, body))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    save = "--save" in sys.argv
    keyword = args[0] if args else ""

    os.chdir(BASE_DIR)  # parquet 使用相对路径
    con = duckdb.connect(database=":memory:")

    setup = open(os.path.join(SQL_DIR, "00_create_views.sql"), encoding="utf-8").read()
    con.execute(setup)

    files = sorted(glob.glob(os.path.join(SQL_DIR, "*.sql")))
    files = [f for f in files if not os.path.basename(f).startswith("00_")]
    if keyword:
        files = [f for f in files if keyword in os.path.basename(f)]

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)

    for path in files:
        fname = os.path.basename(path)
        print("\n" + "=" * 100)
        print(f"■ {fname}")
        print("=" * 100)
        text = open(path, encoding="utf-8").read()
        for name, sql in split_queries(text):
            print(f"\n──【{name}】" + "─" * max(0, 80 - len(name) * 2))
            try:
                df = con.execute(sql).fetchdf()
            except Exception as e:
                print(f"  ❌ 执行失败: {e}")
                continue
            print(df.to_string(index=False, max_rows=40))
            if save:
                safe = re.sub(r"[^\w一-龥]+", "_", name)[:60]
                df.to_csv(os.path.join(OUT_DIR, f"{fname[:2]}_{safe}.csv"),
                          index=False, encoding="utf-8-sig")
    con.close()


if __name__ == "__main__":
    main()
