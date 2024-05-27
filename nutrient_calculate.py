import pandas as pd


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
