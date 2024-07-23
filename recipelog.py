import hashlib
import json
import os
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
    current_time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # この食事記録のためのハッシュ値を生成
    combined_input = f"{username}_{current_time_str}"
    hash_object = hashlib.sha1(combined_input.encode())
    hash_hex = hash_object.hexdigest()

    record_time = datetime.combine(date_input, time_input).strftime("%Y-%m-%d_%H-%M-%S")

    directory = f"records/{username}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_path = f"{directory}record_{hash_hex}.json"
    filename = f"image_{hash_hex}.png"
    image_path = os.path.join(directory, filename)
    image_file.save(image_path)

    result_data = {
        "record_time": record_time,
        "username": username,
        "image": {
            "filename": filename,
            "path": image_path,
        },
        "method": method,
        "ingre_ids": ingre_ids,
        "ingre_names": ingre_names,
        "weights": weights,
        "click_dict": click_dict,
        "used_time": (current_time - start_time).total_seconds(),
        "current_time": current_time_str,
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result_data, file, ensure_ascii=False, indent=4)

    # このユーザの食事履歴のリストを更新
    food_record = os.path.join(directory, "record.json")
    if os.path.exists(food_record):
        with open(food_record, "r", encoding="utf-8") as file:
            records = json.load(file)
    else:
        records = {}

    records[hash_hex] = {
        "record_time": record_time,  # いつの食事記録か？
        "create_time": current_time_str,  # 食事記録が生成された時間
        "edit_time": current_time,  # 食事記録が編集された時間
        "active": True,  # 現在も使われているかどうか
        "duplicate_from": None,  # 元の食事記録はいつのものか？
    }
    with open(food_record, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=4)
