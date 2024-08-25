import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
import streamlit as st
from plotly import express as px

from environment_calculate import get_environment_df
from locales.locale import generate_localer
from nutrient_calculate import (
    append_sum_row_label,
    calc_pfc,
    get_nutri_df_from_food_dict,
    get_nutrient_fact_from_excel,
    get_percent_df,
)
from utils import debug_print


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


def render_ingredient_selectors(
    column,
    session_state_prefix: str,
    label_to_id_and_names: dict,
    predict_ingres: list[int],
    initial_selected_options: list[int] = [],
):
    """
    session_state_prefix: この関数の副作用を抑えるために, session_stateのprefixを指定する.
    column: st.columnsで作成したオブジェクトやst
    initial_selected_options: 記録を編集するときなどに, 記録にある食材を初期選択状態にするためのリスト
    """
    l = generate_localer(st.session_state.lang)
    selected_options_key = session_state_prefix + "_selected_options"
    if selected_options_key not in st.session_state:
        st.session_state[selected_options_key] = initial_selected_options.copy()
    for item in st.session_state[selected_options_key]:
        column.checkbox(
            label_to_id_and_names[int(item) + 1][
                "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"
            ],
            value=True,
            key=item,
            on_change=lambda x: st.session_state[selected_options_key].remove(x),
            args=(item,),
        )

    for item in predict_ingres:
        if item in st.session_state[selected_options_key]:
            continue
        column.checkbox(
            label_to_id_and_names[int(item) + 1][
                "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"
            ],
            value=False,
            key=item,
            on_change=lambda x: st.session_state[selected_options_key].append(x),
            args=(item,),
        )

    def multiselect_on_change():
        name_to_label = {
            item["ja_abbr" if st.session_state.lang == "ja" else "en_abbr"]: key
            for key, item in label_to_id_and_names.items()
        }
        for item in st.session_state["not_in_list_multiselect"]:
            label = int(name_to_label[item]) - 1
            if label in st.session_state[selected_options_key]:
                continue
            st.session_state[selected_options_key].append(label)
        st.session_state["not_in_list_multiselect"] = []
        st.session_state.click_dict["input_text"] = len(
            st.session_state["not_in_list_multiselect"]
        )  # meaningless any more

    # Search box
    column.multiselect(
        l("リストにない食材を検索:"),
        [
            item[1]["ja_abbr" if st.session_state.lang == "ja" else "en_abbr"]
            for item in label_to_id_and_names.items()
        ],
        key="not_in_list_multiselect",
        on_change=multiselect_on_change,
    )
    return st.session_state[selected_options_key].copy()


def render_weight_input(
    column,
    label_to_id_and_names: dict,
    selected_options: list[int],
    initial_weights: list[float] | None = None,
):
    """
    column: st.columnsで作成したオブジェクトやst
    initial_weights: 記録を編集するときなどに, 記録にある食材の重量を初期選択状態にするためのリスト
    """
    l = generate_localer(st.session_state.lang)
    ingre_id_to_weights = get_json_from_file("Labels/weight_median20240615.json")
    weights = []

    column.write(l("一食分に使った量は何グラムですか？"))
    with column.container(height=200):
        for i, label in enumerate(selected_options):
            label_id = int(label) + 1
            ingre_id = label_to_id_and_names[label_id]["id"]
            ingre_name = label_to_id_and_names[label_id][
                "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"
            ]
            median_weight = ingre_id_to_weights[str(ingre_id)][2]

            value = round(float(median_weight), 1)
            min_value = 0.0
            max_value = value * 2
            step = 0.1
            if value > 10:
                value = round(value)
                max_value = int(max_value)
                min_value = int(0)
                step = 1
            else:
                value = float(value)
                min_value = 0.0
                step = 0.1
            if initial_weights is not None:
                value = initial_weights[i]
            weights.append(
                column.slider(ingre_name[:40], min_value, max_value, value, step=step)
            )
    return weights.copy()


def render_meal_info_tabs(
    food_label_amount_unit: list[dict], necessary_nutrients_per_meal: dict
) -> tuple[
    plotly.graph_objs._figure.Figure,
    plotly.graph_objs._figure.Figure,
    pd.DataFrame,
    plotly.graph_objs._figure.Figure,
]:
    """
    food_label_amount_unitなどの食材と重量の情報から, 主要栄養素, PFCバランス, 栄養成分表, 環境への影響の情報を表示する, 副作用のない関数.
    各タブの処理は今後分割したほうが良いかもしれない.
    """
    l = generate_localer(st.session_state.lang)
    nutrients_df = get_nutri_df_from_food_dict(food_label_amount_unit)
    main_nutri_tab, pfc_tab, all_nutri_table_tab, environment_tab = st.tabs(
        [l("主要栄養素"), l("PFCバランス"), l("栄養成分表"), l("環境への影響")]
    )
    with main_nutri_tab:
        percent_df = get_percent_df(nutrients_df, **necessary_nutrients_per_meal)
        percent_df[l("主要栄養素")] = [
            l("カロリー"),
            l("たんぱく質"),
            l("脂質"),
            l("炭水化物"),
            l("塩分"),
        ]
        percent_fig = px.bar(
            percent_df,
            x=l("主要栄養素"),
            y=percent_df.columns[1:].tolist(),
            color_discrete_sequence=px.colors.qualitative.Plotly,
        ).update_layout(
            yaxis_title=l("1食の目安量に対する割合 (%)"),
            height=300,
            legend=dict(itemwidth=30),
        )
        for trace in percent_fig.data:
            raw_series = nutrients_df[trace.name]
            raw_series = raw_series.apply(lambda x: f"{x:.1f}").str.cat(
                ["kcal", "g", "g", "g", "g"], sep=" "
            )
            trace["customdata"] = raw_series
            trace["hovertemplate"] = (
                f"{trace.name}<br>" + "%{customdata}<br>%{y:.1f}%<extra></extra>"
            )
        percent_fig.add_hline(y=100.0, line_color="red", line_dash="dash", line_width=1)
        main_nutri_tab.plotly_chart(percent_fig, use_container_width=True)
    with pfc_tab:
        pfc_df = calc_pfc(nutrients_df)
        percent_fig2 = px.pie(
            values=pfc_df.tolist(),
            names=["Protain", "Fat", "Carb"],
            height=350,
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        pfc_tab.plotly_chart(percent_fig2, use_container_width=True, sort=False)
        pfc_tab.html(
            l(
                "<b>PFCバランスとは？</b><br>三大栄養素であるタンパク質、脂質、炭水化物の摂取バランス。タンパク質は13～20%、脂質は20～30%、炭水化物は50～65%がよいとされています。<br>※20～39歳男女の目標<br>資料：厚生労働省「日本人の食事摂取基準（2020年版）」<br>"
            )
        )
    with all_nutri_table_tab:
        nutrition_fact = get_nutrient_fact_from_excel()
        ingre_ids = [food["ingre_id"] for food in food_label_amount_unit]
        data_df = nutrition_fact.copy()
        data_df = data_df.loc[ingre_ids]
        data_df = data_df.drop(["index", "JName"], axis=1)
        data_df.insert(1, "weight", 0)

        for ii in range(len(data_df)):
            weight = food_label_amount_unit[ii]["amount"]
            data_df.iloc[ii, 1] = weight
            for jj in range(2, len(data_df.columns)):
                if not math.isnan(data_df.iloc[ii, jj]):
                    data_df.iloc[ii, jj] = float(data_df.iloc[ii, jj]) * weight / 100
        append_sum_row_label(data_df)
        all_nutri_table_tab.dataframe(data_df, width=800)
    with environment_tab:
        env_dataset_df = get_environment_df()
        food_env_df = pd.DataFrame(
            {l("環境への影響"): [l("TMR係数"), "GWP", l("反応性窒素")]}
        )
        for food in food_label_amount_unit:
            food_id = food["ingre_id"]
            food_env_df[food["canonical_name"]] = (
                env_dataset_df.loc[food_id].values * float(food["amount"]) / 1000
            )  # TODO: 現在はamountがgだが, 今後g意外も選択できるようになったら修正
        debug_print("food_env_df", food_env_df)
        environment_fig = px.bar(
            food_env_df,
            x=l("環境への影響"),
            y=food_env_df.columns[1:].tolist(),
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        environment_tab.plotly_chart(environment_fig, use_container_width=True)
        environment_tab.html(
            l(
                "TMR係数 (Total Material Requirement)とは, その食品を提供するために必要な採掘活動量です"
            )
        )
        environment_tab.html(
            l("GWP (Global Warming Potential)とは, 温室効果ガスの排出量です")
        )
        environment_tab.html(l("反応性窒素とは, 窒素酸化物の排出量です"))
    return percent_fig, percent_fig2, data_df, environment_fig


def save_results(
    username,
    image_file,
    method,
    ingre_ids,
    ingre_names,
    weights,
    meal_time,
    click_dict,
    start_time,
    main_nutri_fig,
    pfc_fig,
    detail_nutri_df,
    environment_fig,
) -> int:
    """
    Save the results of the dish.
    Return the dish number.
    """
    # 「保存」を実行した時刻
    current_time = datetime.now()
    current_time_str = current_time.isoformat(timespec="seconds")
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

    with (dish_dir_path / (current_time_str + "_main_nutri_fig.json")).open(
        mode="w", encoding="utf-8"
    ) as f:
        f.write(pio.to_json(main_nutri_fig))
    with (dish_dir_path / (current_time_str + "_pfc_fig.json")).open(
        mode="w", encoding="utf-8"
    ) as f:
        f.write(pio.to_json(pfc_fig))
    with (dish_dir_path / (current_time_str + "_environment_fig.json")).open(
        mode="w", encoding="utf-8"
    ) as f:
        f.write(pio.to_json(environment_fig))

    detail_nutri_df.to_csv(dish_dir_path / (current_time_str + "_detail_nutri.csv"))
    return dish_count
