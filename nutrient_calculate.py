import json

import pandas as pd
import streamlit as st


@st.cache_data
def get_nutri_df_from_food_dict(food_label_amount_unit: dict):
    """
    Get nutrient dataframe from food dict
    """
    with open("Labels/nutrition_dic.json", "r", encoding="utf-8") as file:
        nutrients_infos = json.load(file)
    unit_trans_csv = pd.read_csv("Labels/weight_trans.csv")
    nutrient = {
        "主要栄養素": [
            "カロリー (kcal)",
            "たんぱく質 (g)",
            "脂質 (g)",
            "炭水化物 (g)",
            "塩分 (g)",
        ],
    }
    nutrient_codes = ["ENERC_KCAL", "PROT-", "FAT-", "CHOAVLM", "NACL_EQ"]

    for food_item in food_label_amount_unit:
        amount = food_item["amount"]
        unit = food_item["unit"]
        canonical_name = food_item["canonical_name"]
        ingre_id = food_item["ingre_id"]

        to_gram = (
            unit_trans_csv.loc[unit_trans_csv["食品番号"] == int(ingre_id), unit].iloc[
                0
            ]
            if unit != "g"
            else 1
        )
        to_gram = max(to_gram, 0)
        weight = amount * to_gram

        nutri_list = []
        for code in nutrient_codes:
            try:
                nutri_list.append(float(nutrients_infos[ingre_id][code]) * weight / 100)
            except ValueError:
                nutri_list.append(float(0))
        nutrient[canonical_name] = nutri_list

    nutrients_df = pd.DataFrame(nutrient)
    nutrients_df["主要栄養素"] = (
        pd.Categorical(  # walk-around needed for auto-sorting bug
            nutrients_df["主要栄養素"],
            categories=[
                "カロリー (kcal)",
                "たんぱく質 (g)",
                "脂質 (g)",
                "炭水化物 (g)",
                "塩分 (g)",
            ],
            ordered=True,
        )
    )
    return nutrients_df


def get_necessary_calories(sex: str, age: int, physical_activity_level: int) -> int:
    calorie_data = pd.read_csv("Labels/necessary_nutrients/calories.csv")
    filtered_data = calorie_data[
        (calorie_data["age_start"] <= age) & (calorie_data["age_end"] >= age)
    ]
    if filtered_data.empty:
        return None
    column_name = f"{sex}_level_{physical_activity_level}"
    required_calories = filtered_data[column_name].values[0]
    return required_calories


def get_necessary_protein(sex: str, age: int) -> int:
    protein_data = pd.read_csv("Labels/necessary_nutrients/protein.csv")
    filtered_data = protein_data[
        (protein_data["age_start"] <= age) & (protein_data["age_end"] >= age)
    ]
    if filtered_data.empty:
        return None
    necessary_protein = filtered_data[sex].values[0]
    return necessary_protein


def calculate_necessary_nutrients(sex: str, age: int, physical_activity_level: int):
    """
    Calculate necessary nutrients.
    Return dictionary with keys 'kcal', 'protein', 'fat', 'carb', 'salt'.
    If some data is missing, return None.
    """
    CARB_CALORIE_RATIO = 0.575
    FAT_CALORIE_RATIO = 0.25
    CARB_CALORIE_PAR_GRAM = 4
    FAT_CALORIE_PAR_GRAM = 9
    NECESSARY_SALT = 6  ## TODO: rethink this value
    necessary_calories = get_necessary_calories(sex, age, physical_activity_level)
    necessary_protein = get_necessary_protein(sex, age)
    if necessary_calories is None or necessary_protein is None:
        return None
    necessary_fat = necessary_calories * FAT_CALORIE_RATIO / FAT_CALORIE_PAR_GRAM
    necessary_carb = necessary_calories * CARB_CALORIE_RATIO / CARB_CALORIE_PAR_GRAM
    return {
        "kcal": necessary_calories,
        "protein": necessary_protein,
        "fat": necessary_fat,
        "carb": necessary_carb,
        "salt": NECESSARY_SALT,
    }
