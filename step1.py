import json
import os
from datetime import datetime
from enum import IntEnum

# import japanese_clip as ja_clip
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
from PIL import Image
from plotly import express as px

import nutrient_calculate
from src.dataloader import VireoLoader
from src.model_clip import Recognition

DEBUG = True


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


@st.cache_data
def get_ingres_name():
    file_path = "Labels/IngredientList588.txt"
    ingres = []
    name2idx = {}
    i = 0
    with open(file_path, "r") as file:
        for line in file:
            line = line.rstrip("\n")
            ingres.append(line)
            name2idx[line] = i
            i += 1
    return ingres, name2idx


@st.cache_data
def get_canonical_ingres_name():
    """
    Label -> Canonical Name
    ex) 1: 砂糖
    3(NA, 水) is excluded
    """
    label_to_canonical_name = {}
    with open("Labels/foodid_dict.json", "r", encoding="utf-8") as file:
        ingredient_dict = json.load(file)
    id_to_canonical_name = {}
    for record in pd.read_csv("Labels/ingre_id_label_expression_table.csv").to_dict(
        "records"
    ):
        id_to_canonical_name[record["ingredientid"]] = record["ingredient_exp2"]
    for label, info_list in ingredient_dict.items():
        if label == "3":
            continue
        ingredient_id = int(info_list[0])
        if ingredient_id in id_to_canonical_name:
            label_to_canonical_name[label] = id_to_canonical_name[ingredient_id]
        else:
            label_to_canonical_name[label] = info_list[2]
    return label_to_canonical_name


@st.cache_data
def get_normalized_co_occurrence_matrix():
    co_occurrence_matrix = np.load("Labels/co_occurrence_matrix.npy", allow_pickle=True)
    row_sums = co_occurrence_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = co_occurrence_matrix / row_sums
    return normalized_matrix


@st.cache_data
def relative_pos():
    relative_pos = np.load("Labels/relative_pos.npy", allow_pickle=True)
    flipped_relative_pos = 1 - relative_pos
    return flipped_relative_pos


@st.cache_data
def get_pos_probability(text_probs):
    alpha = 0.1
    flipped_relative_pos = relative_pos()
    text_probs = np.array(text_probs)
    flipped_relative_pos = np.array(flipped_relative_pos)
    final_probs = (1 - alpha) * text_probs + alpha * flipped_relative_pos
    # st.write("text_probs:",text_probs)
    # st.write("flipped_relative_pos",flipped_relative_pos)
    return final_probs.tolist()


@st.cache_data
def get_ingre_probability(ingres, uploaded_image):
    # Load CLIP model
    device = "cpu"
    model, preprocess = ja_clip.load("rinna/japanese-cloob-vit-b-16", device=device)
    tokenizer = ja_clip.load_tokenizer()

    input_image = preprocess(uploaded_image).unsqueeze(0).to(device)
    search = model.get_image_features(input_image).cpu()

    encodings = ja_clip.tokenize(
        texts=[f"{ing}を使った料理" for ing in ingres],
        max_seq_len=77,
        device=device,
        tokenizer=tokenizer,  # this is optional. if you don't pass, load tokenizer each time
    )

    with torch.no_grad():
        ingre_text_features = model.get_text_features(**encodings)
        image_features = search.to(device)
        # text_probs = (100.0 * image_features @ ingre_text_features.T).softmax(dim=-1)
        text_probs = 100.0 * image_features @ ingre_text_features.T
        # values, indices = text_probs.topk(10)
        # for item in indices[0]:
        #     res_label.append(item.cpu().numpy().astype(int))

    return text_probs


@st.cache_data
def get_ingre_prob_from_model(uploaded_image):
    model_path = "Models/clip_v32_epoch21_20.938.pth"

    device = torch.device("cpu")
    model = Recognition()
    model = nn.DataParallel(model)
    model.to(device)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=device))
    # model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    criterions = [
        nn.CrossEntropyLoss().to("cpu"),
        nn.BCELoss(reduction="none").to("cpu"),
    ]

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    test_loader = torch.utils.data.DataLoader(
        VireoLoader(
            uploaded_image,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=1,
        shuffle=True,
        timeout=1000,
        num_workers=1,
        # pin_memory=True,
    )

    for i, inputs in enumerate(test_loader):
        imgs = inputs[0].to("cpu")
        outputs = model(imgs)

    text_probs = outputs[1].cpu()
    # text_probs = torch.from_numpy(np.load('/home/l_wang/vireofood251/RA-CLIP/ingre_feature_pasta.npy'))
    min_value = torch.min(text_probs)
    abs_min_value = torch.abs(min_value)
    normalized_tensor = text_probs + abs_min_value
    # to positive

    min_normalized_value = torch.min(normalized_tensor)
    max_normalized_value = torch.max(normalized_tensor)
    normalized_tensor = (normalized_tensor - min_normalized_value) / (
        max_normalized_value - min_normalized_value
    )
    # 0-1

    return normalized_tensor


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
def update_mask_method1(selected_items, mask):
    for selected in selected_items:
        mask[selected] = 0
    return mask


@st.cache_data
def get_current_candidate(candidate_nums, uploaded_image, mask):
    text_probs = get_ingre_prob_from_model(uploaded_image)
    probability_scores = [item for sublist in text_probs.tolist() for item in sublist]
    pos_probability = get_pos_probability(probability_scores)
    cur_prob = pos_probability * mask
    top_k_indices = np.argsort(cur_prob)[-candidate_nums:][::-1]
    return top_k_indices.tolist()


@st.cache_data
def get_json_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


@st.cache_data
def get_percent_df(df, kcal, protein, fat, carb, salt):
    percent_df = df.copy()
    percent_df["target"] = [kcal, protein, fat, carb, salt]
    percent_df.iloc[:, 1:] = (
        percent_df.iloc[:, 1:].div(percent_df["target"], axis=0) * 100
    )
    percent_df.drop("target", axis=1, inplace=True)
    print(percent_df)
    percent_df["主要栄養素"] = ["カロリー", "たんぱく質", "脂質", "炭水化物", "塩分"]
    percent_df = percent_df.round(2)
    return percent_df


def save_results(username, image_file, method, ingredients, ingres_convert, click_dict):
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    end_time = datetime.now()
    time_difference = end_time - st.session_state.start_time
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
        ingre_names.append(ingres_convert[int(item)])

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


class StreamlitStep(IntEnum):
    SESSION_WHILE_INIT = 0
    WAIT_FOR_IMAGE = 1
    IMAGE_UPLOADED = 2
    WAIT_FOR_INGREDIENT_SELECTION = 3
    AFTER_INGREDIENT_SELECTION_INIT = 4
    WAIT_FOR_AMOUNT_INPUT = 5
    FINISH = 6

    def __str__(self):
        return self.name


def page_1():
    st.title("材料リストによる食事管理")
    debug_print(st.session_state)

    if st.session_state.stage == StreamlitStep.SESSION_WHILE_INIT:
        st.session_state.stage = StreamlitStep.WAIT_FOR_IMAGE
        st.session_state.click_dict = {"button": 0, "checkbox": 0, "input_text": 0}

    def change_stage_to_image_uploaded():
        st.session_state.stage = StreamlitStep.IMAGE_UPLOADED

    uploaded_image = st.file_uploader(
        "食事の写真をアップロードしてください",
        type=["jpg", "jpeg", "png"],
        on_change=change_stage_to_image_uploaded,
    )

    if st.session_state.stage <= StreamlitStep.WAIT_FOR_IMAGE:
        return

    c1, c2, c3 = st.columns((1, 1, 1))  # visual statements
    if uploaded_image:
        c1.image(uploaded_image)
        image = Image.open(uploaded_image)

    candidate_nums = 10  # stateless variables
    label_to_canonical_name = get_canonical_ingres_name()
    canonical_name_to_label = {v: k for k, v in label_to_canonical_name.items()}

    if (
        st.session_state.stage == StreamlitStep.IMAGE_UPLOADED
    ):  # initializations should be done before next stage
        st.session_state.stage = StreamlitStep.WAIT_FOR_INGREDIENT_SELECTION
        st.session_state.start_time = datetime.now()

        mask = [1] * 588
        mask[2] = 0
        mask = np.array(mask)
        st.session_state.mask = mask
        st.session_state.selected_options = {}

        st.session_state.predict_ingres = get_current_candidate(
            candidate_nums,
            uploaded_image,
            st.session_state.mask,
        )
    if st.session_state.stage <= StreamlitStep.IMAGE_UPLOADED:
        return

    c2.write("食材候補：料理に含まれている材料をチェックしてください")
    for item in st.session_state.predict_ingres:
        st.session_state.selected_options[item] = c2.checkbox(
            label_to_canonical_name[str(int(item) + 1)]
        )
    not_in_list_multiselect = c2.multiselect(
        "リストにない食材を検索:", label_to_canonical_name.values(), []
    )

    st.session_state.selected_ingres = [  # 0-indexed int label
        item for item, selected in st.session_state.selected_options.items() if selected
    ]
    if not_in_list_multiselect:
        for name in not_in_list_multiselect:
            st.session_state.selected_ingres.append(
                int(canonical_name_to_label[name]) - 1
            )
        st.session_state.click_dict["input_text"] = len(not_in_list_multiselect)

    st.session_state.mask = update_mask(
        st.session_state.selected_ingres,
        st.session_state.mask,
    )

    if c2.button("新しい食材候補を生成する"):
        st.session_state.click_dict["button"] += 1
        st.session_state.mask = update_mask_method1(
            st.session_state.predict_ingres, st.session_state.mask
        )  # delete other unselected ingredients

        st.session_state.predict_ingres = get_current_candidate(
            candidate_nums,
            uploaded_image,
            st.session_state.mask,
        )

        debug_print("generate:", st.session_state.predict_ingres)
        st.rerun()

    c3.divider()
    c3.write("選択された食材リスト:")
    for item in st.session_state.selected_ingres:
        checkbox_value = c3.checkbox(
            label_to_canonical_name[str(int(item) + 1)], value=True
        )
        if not checkbox_value:
            st.session_state.selected_options[item] = False
    c3.divider()

    if c3.button("完了"):
        st.session_state.stage = StreamlitStep.AFTER_INGREDIENT_SELECTION_INIT

    if st.session_state.stage <= StreamlitStep.WAIT_FOR_INGREDIENT_SELECTION:
        return

    st.session_state.click_dict["checkbox"] = (
        len(st.session_state.selected_ingres)
        - st.session_state.click_dict["input_text"]
    )

    if st.session_state.stage == StreamlitStep.AFTER_INGREDIENT_SELECTION_INIT:
        save_results(
            st.session_state.username,
            image,
            "method_2",
            st.session_state.selected_ingres,
            list(label_to_canonical_name.values()),
            st.session_state.click_dict,
        )
        st.session_state.stage = StreamlitStep.WAIT_FOR_AMOUNT_INPUT
    c3.success("回答を記録しました！次のページに進んでください。")

    foodid_dic = get_json_from_file(
        "Labels/foodid_dict.json"
    )  # {"1": [id, exp, label_name, [whole_exps]]}
    ingre_id_to_weights = get_json_from_file("Labels/weight_median.json")

    ingre_names = []
    ingre_exps = []
    ingre_ids = []
    median_weights = []
    for item in st.session_state.selected_ingres:
        label_id = int(item) + 1
        ingre_id = foodid_dic[str(label_id)][0]
        ingre_ids.append(ingre_id)
        ingre_names.append(label_to_canonical_name[str(label_id)])
        ingre_exps.append(foodid_dic[str(label_id)][1])
        median_weights.append(ingre_id_to_weights[ingre_id][1])

    data = pd.DataFrame(
        {
            "ingredients": ingre_names,
            "standard_exp": ingre_exps,
            "amount": median_weights,
            "unit": ["g"] * len(st.session_state.selected_ingres),
        }
    )
    data["index"] = data.index + 1
    data.set_index("index", inplace=True)

    data_df = st.data_editor(
        data,
        column_config={
            "index": st.column_config.Column("index"),
            "ingredients": "食材名",
            "standard_exp": st.column_config.Column("食品名"),
            "amount": st.column_config.NumberColumn("重さ"),
            "unit": st.column_config.SelectboxColumn(
                "単位",
                help="The category of the unit",
                options=[
                    "g",
                    "無単位",
                    "枚",
                    "本",
                    "個片丁株玉房",
                    "杯/カップ",
                    "半分",
                    "摘/少",
                    "小/小匙",
                    "中",
                    "大/大匙",
                    "一掴",
                    "袋",
                    "箱",
                    "缶/カン",
                    "CC",
                    "cm",
                    "束",
                    "合",
                ],
                required=True,
            ),
        },
        hide_index=False,
    )

    if st.button("入力完了"):
        st.session_state.stage = StreamlitStep.FINISH
    if st.session_state.stage <= StreamlitStep.WAIT_FOR_AMOUNT_INPUT:
        return

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

    ingre_infos = {}
    for i, row in data_df.iterrows():
        item = st.session_state.selected_ingres[i - 1]
        amount = row["amount"]
        unit = row["unit"]

        ingre_id = ingre_ids[i - 1]
        to_gram = (
            unit_trans_csv.loc[unit_trans_csv["食品番号"] == int(ingre_id), unit].iloc[
                0
            ]
            if unit != "g"
            else 1
        )
        to_gram = max(to_gram, 0)
        weight = amount * to_gram

        ingre_infos[ingre_id] = {
            "amount": amount,
            "unit": unit,
            "weight": weight,
        }
        nutri_list = []
        for code in nutrient_codes:
            try:
                nutri_list.append(float(nutrients_infos[ingre_id][code]) * weight / 100)
            except ValueError:
                nutri_list.append(float(0))

        new_ing_dic = {label_to_canonical_name[str(int(item) + 1)]: nutri_list}
        nutrient.update(new_ing_dic)

    total_df = pd.DataFrame(nutrient)
    total_df["主要栄養素"] = pd.Categorical(  # walk-around needed for auto-sorting bug
        total_df["主要栄養素"],
        categories=[
            "カロリー (kcal)",
            "たんぱく質 (g)",
            "脂質 (g)",
            "炭水化物 (g)",
            "塩分 (g)",
        ],
        ordered=True,
    )

    necessary_nutrients = nutrient_calculate.calculate_necessary_nutrients(
        users[st.session_state.username]["sex"],
        users[st.session_state.username]["age"],
        users[st.session_state.username]["physical_activity_level"],
    )
    necessary_nutrients_per_meal = {
        key: value / 3 for key, value in necessary_nutrients.items()
    }
    debug_print("necessary nutrients per meal", necessary_nutrients_per_meal)

    percent_df = get_percent_df(total_df, **necessary_nutrients_per_meal)
    percent_fig = px.bar(
        percent_df, x="主要栄養素", y=percent_df.columns[1:].tolist()
    ).update_layout(yaxis_title="1食の目安量に対する割合 (%)")
    for trace in percent_fig.data:
        raw_series = total_df[trace.name]
        raw_series = raw_series.apply(lambda x: f"{x:.2f}").str.cat(
            ["kcal", "g", "g", "g", "g"], sep=" "
        )
        trace["customdata"] = raw_series
        trace["hovertemplate"] = (
            f"{trace.name}<br>" + "%{customdata}<br>%{y:.2f}%<extra></extra>"
        )
    percent_fig.add_hline(y=100.0, line_color="red", line_dash="dash", line_width=1)
    st.plotly_chart(percent_fig)
    st.write("あなたの1食あたりの目標栄養摂取量は")
    st.write(f"カロリー {necessary_nutrients_per_meal['kcal']:.2f} kcal")
    st.write(f"たんぱく質 {necessary_nutrients_per_meal['protein']:.2f} g")
    st.write(f"脂質 {necessary_nutrients_per_meal['fat']:.2f} g")
    st.write(f"炭水化物 {necessary_nutrients_per_meal['carb']:.2f} g")
    st.write(f"塩分 {necessary_nutrients_per_meal['salt']:.2f} g です")


users = json.load(open("userdata/users.json", "r"))


def authenticate(username, password):
    if username not in users:
        return False
    if users[username]["password"] == password:
        return True
    else:
        return False


def main():
    st.set_page_config(
        page_title="RecipeLog2023",
        page_icon=":curry:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "stage" not in st.session_state:
        st.title("Login")
        c1, _, _ = st.columns((1, 1, 1))
        username = c1.text_input(
            "アカウント:",
        )
        password = c1.text_input("パスワード:", type="password")

        if c1.button("Login"):
            if not authenticate(username, password):
                c1.error("アカウント／パスワードが正しくありません")
            else:
                st.session_state.username = username
                st.session_state.register = True
                st.session_state.stage = StreamlitStep.SESSION_WHILE_INIT
                st.rerun()
    else:
        page_1()


if __name__ == "__main__":
    main()
