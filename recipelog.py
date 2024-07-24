import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def get_label_to_id_and_names():
    """
    Water (3) is excluded from the list
    label is 1-indexed
    """
    jpn_eng_expression_df = pd.read_csv("Labels/foodexpList20240612.csv")
    label_to_id_and_names = {}
    for record in jpn_eng_expression_df.to_dict("records"):
        label_to_id_and_names[int(record["ingredient_label"])] = {
            "id": int(record["foodid"]),
            "ja_abbr": record["JPN Abbr"],
            "ja_full": record["JPN full"],
            "en_abbr": record["ENG Abbr"],
            "en_full": record["ENG full"],
        }
    return label_to_id_and_names


@st.cache_data
def get_normalized_co_occurrence_matrix():
    co_occurrence_matrix = np.load("Labels/co_occurrence_matrix.npy", allow_pickle=True)
    row_sums = co_occurrence_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = co_occurrence_matrix / row_sums
    return normalized_matrix


def sigmoid(x, a):
    return 1.0 / (1.0 + np.exp(-a * x))


@st.cache_data
def update_mask(selected_items, mask):
    """
    処理1:
    selected_itemsとの共起の確率が低いものはマスクする.
    thresholdでどれくらいの確率からマスクするかをコントロールできる.
    処理2:
    選択した食材につき, シグモイド関数に1+normalized_matrixを入力しマスクを更新することを繰り返す.
    最後はマスクの合計が588になるよう正規化する.

    複数回同じselected_itemsを入力すると, 初回のみmaksの更新が行われるような実装になっている.
    """
    normalized_matrix = get_normalized_co_occurrence_matrix()
    threshold = 0.0
    normalized_matrix = np.where(
        normalized_matrix < threshold / 100, 0, normalized_matrix
    )
    for selected in selected_items:
        if mask[selected] != 0.0:
            # print(selected)
            # print(sigmoid(normalized_matrix[selected] + 1, 1)) # [0.73483725 0.73735383 0.74200039...]
            mask[selected] = 0.0
            mask = mask * sigmoid(normalized_matrix[selected] + 1, 1)
    mask = mask / np.sum(mask) * 588
    return mask


@st.cache_data
def get_json_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def save_results(
    username,
    image_file,
    method,
    ingre_ids,
    ingre_names,
    weights,
    date_input,
    time_input,
    click_dict,
    start_time,
):
    # 「保存」を実行した時刻
    current_time = datetime.now()
    current_time_str = current_time.isoformat(timespec="seconds")
    meal_time = datetime.combine(date_input, time_input)
    meal_time_str = meal_time.isoformat(timespec="minutes")

    user_dir_path = Path(f"records/{username}")
    if not user_dir_path.exists():
        user_dir_path.mkdir()

    meal_dir_path = user_dir_path / meal_time_str
    if not meal_dir_path.exists():
        meal_dir_path.mkdir()

    dish_count = len(list(meal_dir_path.glob("*")))
    dish_dir_path = meal_dir_path / str(dish_count)
    if not dish_dir_path.exists():
        dish_dir_path.mkdir()

    result_data = {
        "method": method,
        "ingre_ids": ingre_ids,
        "ingre_names": ingre_names,
        "weights": weights,
        "click_dict": click_dict,
        "used_time": (current_time - start_time).total_seconds(),
    }
    with (dish_dir_path / (current_time_str + ".json")).open(
        mode="w", encoding="utf-8"
    ) as f:
        json.dump(result_data, f, ensure_ascii=False)
    image_file.save(dish_dir_path / (current_time_str + ".jpg"))
