import os
import numpy as np
import json
import pandas as pd
from datetime import datetime
import streamlit as st

DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

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
def get_name_to_label(label_to_id_and_names):
    name_to_label = {}
    for k in label_to_id_and_names.keys():
        name_to_label[label_to_id_and_names[k][
            "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"]
                      ] = k
    return name_to_label


@st.cache_data
def get_normalized_co_occurrence_matrix():
    co_occurrence_matrix = np.load("Labels/co_occurrence_matrix.npy", allow_pickle=True)
    row_sums = co_occurrence_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = co_occurrence_matrix / row_sums
    return normalized_matrix


@st.cache_data
def update_mask(selected_items, mask):
    normalized_matrix = get_normalized_co_occurrence_matrix()
    threshold = 0.5
    normalized_matrix = np.where(
        normalized_matrix < threshold / 100, 0, normalized_matrix
    )
    for selected in selected_items:
        distances = normalized_matrix[selected]
        dist_mask = np.where(distances == 0, 0, 1)
        mask = dist_mask & mask
    return mask

@st.cache_data
def get_json_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_results(
    username, image_file, method, ingredients, ingres_convert, click_dict, start_time
):
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    end_time = datetime.now()
    time_difference = end_time - start_time
    directory = f"Results/{username}/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    existing_files = [
        file
        for file in os.listdir(directory)
        if file.startswith("result") and file.endswith(".json")
    ]
    next_serial_number = len(existing_files) + 1

    output_path = f"{directory}result_{next_serial_number}_{method}_{current_time}.json"

    ingre_names = []
    for item in ingredients:
        ingre_names.append(ingres_convert[int(item) + 1])

    filename = f"image_{next_serial_number}.png"
    image_path = os.path.join(directory, filename)
    debug_print(image_path)
    image_file.save(image_path)

    result_data = {
        "username": username,
        "image": {
            "filename": filename,
            "path": image_path,
        },
        "method": method,
        "ingredients": ingredients,
        "ingre_names": ingre_names,
        "click_dict": click_dict,
        "used_time": time_difference.total_seconds(),
        "current_time": current_time,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result_data, file, ensure_ascii=False, indent=4)
