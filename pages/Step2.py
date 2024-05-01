import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
from PIL import Image

from src.dataloader import VireoLoader
from src.model_clip import Recognition


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
def get_ingre_prob_from_model(uploaded_image):
    # model_path = '/home/l_wang/vireofood251/Compare_subdatasets/checkpoints_region/Model/clip_vit32/n_epoch21_20.938.pth'
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
def update_mask(selected_items, mask, normalized_matrix):
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
def get_current_candidate(candidate_nums, flat_list, mask, normalized_matrix):
    cur_prob = flat_list * mask
    top_k_indices = np.argsort(cur_prob)[-candidate_nums:][::-1]
    return top_k_indices.tolist()


@st.cache_data
def get_current_candidate_method1(candidate_nums, flat_list, mask, normalized_matrix):
    cur_prob = flat_list * mask
    top_k_indices = np.argsort(cur_prob)[-candidate_nums:][::-1]
    return top_k_indices.tolist()


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
    print(image_path)
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


def set_state(i):
    st.session_state.stage = i
    # st.session_state.click_dict["button"] += 1


st.set_page_config(
    page_title="RecipeLog2023",
    page_icon=":curry:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "username" not in st.session_state:
    st.write("ページを再度開いてください。")
    newpage_url = "http://163.220.177.123/l_wang/DAI/"
    st.link_button("新しページへ", newpage_url)

username = st.session_state.username
predict_method = st.session_state.predict_method
if "update_session_state" not in st.session_state:
    st.session_state.clear()
    st.session_state.update_session_state = True

if "ingre_finish" not in st.session_state:
    st.session_state.ingre_finish = False

if "dataframe_finish" not in st.session_state:
    st.session_state.dataframe_finish = False

if "init_prediction" not in st.session_state:
    st.session_state.init_prediction = False

if "init_1" not in st.session_state:
    st.session_state.init_1 = True
    st.session_state.username = username
    st.session_state.predict_method = predict_method

    st.session_state.click_dict = {"button": 0, "checkbox": 0, "input_text": 0}
    st.session_state.stage = 0
    st.session_state.ingres, st.session_state.name2idx = get_ingres_name()
    st.session_state.uploaded_image = None
    st.session_state.saved = False

st.title("材料リストによる食事管理")

if st.session_state.uploaded_image is None:
    uploaded_image = st.file_uploader(
        "食事の写真をアップロードしてください", type=["jpg", "jpeg", "png"]
    )
    st.session_state.uploaded_image = uploaded_image
else:
    st.file_uploader(
        "食事の写真をアップロードしてください", type=["jpg", "jpeg", "png"]
    )
    uploaded_image = st.session_state.uploaded_image
if uploaded_image and st.session_state.stage == 0 and st.session_state.predict_method:
    st.session_state.stage += 1

methods = ["method_1", "method_2"]
if "method_changed" not in st.session_state:
    st.session_state.method_changed = False

if st.session_state.method_changed == False:
    st.session_state.method_changed = True
    if st.session_state.predict_method == methods[0]:
        st.session_state.predict_method = methods[1]
    else:
        st.session_state.predict_method = methods[0]

# Predict Module
c1, c2, c3 = st.columns((1, 1, 1))
if st.session_state.stage >= 1:
    threshold = 0.5
    candidate_nums = 10
    ingres, name2idx = st.session_state.ingres, st.session_state.name2idx
    # display image
    c1.image(st.session_state.uploaded_image, use_column_width=False, width=150)
    image = Image.open(st.session_state.uploaded_image)

    if "init_2" not in st.session_state:
        st.session_state.init_2 = True
        st.session_state.predict_ingres = []
        st.session_state.selected_ingres = []
        text_probs = get_ingre_prob_from_model(uploaded_image)
        st.session_state.probability_scores = [
            item for sublist in text_probs.tolist() for item in sublist
        ]
        st.session_state.pos_probability = get_pos_probability(
            st.session_state.probability_scores
        )
        normalized_matrix = get_normalized_co_occurrence_matrix()
        st.session_state.normalized_matrix = np.where(
            normalized_matrix < threshold / 100, 0, normalized_matrix
        )
        mask = [1] * 588
        mask[2] = 0
        mask = np.array(mask)
        st.session_state.mask = mask
        st.session_state.selected_options = {}

    if st.session_state.init_prediction == False:
        if st.session_state.predict_method == "method_1":
            st.session_state.predict_ingres = get_current_candidate_method1(
                candidate_nums,
                st.session_state.probability_scores,
                st.session_state.mask,
                st.session_state.normalized_matrix,
            )
        if st.session_state.predict_method == "method_2":
            # st.session_state.predict_ingres = get_current_candidate(candidate_nums, st.session_state.probability_scores, st.session_state.mask, st.session_state.normalized_matrix)
            st.session_state.predict_ingres = get_current_candidate(
                candidate_nums,
                st.session_state.pos_probability,
                st.session_state.mask,
                st.session_state.normalized_matrix,
            )
        st.session_state.init_prediction = True

st.session_state.stage += 1

if st.session_state.stage >= 2:
    c2.write("食材候補：料理に含まれている材料をチェックしてください")
    print("now:", st.session_state.predict_ingres)

    if "start_time" not in st.session_state:
        st.session_state.start_time_flag = False
    if st.session_state.start_time_flag == False:
        st.session_state.start_time_flag = True
        st.session_state.start_time = datetime.now()

    for item in st.session_state.predict_ingres:
        st.session_state.selected_options[item] = c2.checkbox(
            ingres[int(item)]
        )  # key=f'ingre_{item}')

    ingres_without_NA = ingres[:2] + ingres[3:]
    options = c2.multiselect("リストにない食材を検索:", ingres_without_NA, [])
    st.session_state.selected_ingres = [
        item for item, selected in st.session_state.selected_options.items() if selected
    ]
    # st.session_state.click_dict["checkbox"] += 1
    if options:
        for option in options:
            st.session_state.selected_ingres.append(name2idx[option])
        st.session_state.click_dict["input_text"] = len(options)

    if st.session_state.predict_method == "method_1":
        st.session_state.mask = update_mask_method1(
            st.session_state.selected_ingres, st.session_state.mask
        )
    if st.session_state.predict_method == "method_2":
        st.session_state.mask = update_mask(
            st.session_state.selected_ingres,
            st.session_state.mask,
            st.session_state.normalized_matrix,
        )

    if c2.button("新しい食材候補を生成する"):
        st.session_state.click_dict["button"] += 1
        st.session_state.mask = update_mask_method1(
            st.session_state.predict_ingres, st.session_state.mask
        )  # delete other unselected ingredients

        if st.session_state.predict_method == "method_1":
            st.session_state.predict_ingres = get_current_candidate_method1(
                candidate_nums,
                st.session_state.probability_scores,
                st.session_state.mask,
                st.session_state.normalized_matrix,
            )
        if st.session_state.predict_method == "method_2":
            # st.session_state.predict_ingres = get_current_candidate(candidate_nums, st.session_state.probability_scores, st.session_state.mask, st.session_state.normalized_matrix)
            st.session_state.predict_ingres = get_current_candidate(
                candidate_nums,
                st.session_state.pos_probability,
                st.session_state.mask,
                st.session_state.normalized_matrix,
            )

        print("generate:", st.session_state.predict_ingres)
        st.rerun()

    c3.divider()
    c3.write("選択された食材リスト:")
    # for item in st.session_state.selected_ingres:
    #     c3.checkbox(ingres[int(item)], value=True)

    for item in st.session_state.selected_ingres:
        checkbox_value = c3.checkbox(ingres[int(item)], value=True)
        if not checkbox_value:
            st.session_state.selected_options[item] = False

    c3.divider()

    if c3.button("完了"):  # , on_click=set_state, args=[3]
        st.session_state.ingre_finish = True
        set_state(3)

    if st.session_state.ingre_finish == True:
        st.session_state.stage = 3
        st.session_state.click_dict["checkbox"] = (
            len(st.session_state.selected_ingres)
            - st.session_state.click_dict["input_text"]
        )
        st.session_state.selected_ingres = [
            item
            for item, selected in st.session_state.selected_options.items()
            if selected
        ]
        if st.session_state.saved == False:
            save_results(
                st.session_state.username,
                image,
                st.session_state.predict_method,
                st.session_state.selected_ingres,
                ingres,
                st.session_state.click_dict,
            )
            st.session_state.saved = True
        c3.success("回答を記録しました！アンケートにお進んでください。")

        url = "https://forms.gle/9cGFNxp3VBiyt78f7"
        c3.link_button("アンケートへ", url)

        # if st.session_state.question_finish == True:
        #     new_page = 'http://163.220.177.123/l_wang/DAI/'
        #     c3.link_button("新しい料理登録", new_page)

        with open("Labels/foodid_dict.json", "r", encoding="utf-8") as file:
            foodid_dic = json.load(file)  # {"1": [id, exp, label_name, [whole_exps]]}

        ingre_names = []
        ingre_exps = []
        ingre_ids = []
        for item in st.session_state.selected_ingres:
            label_id = int(item) + 1
            ingre_names.append(foodid_dic[str(label_id)][2])
            ingre_exps.append(foodid_dic[str(label_id)][1])
            ingre_ids.append(foodid_dic[str(label_id)][0])

        if "generate_value" not in st.session_state:
            st.session_state.generate_value = False
        if "edited_amount" not in st.session_state:
            st.session_state.edited_amount = []
        if "edited_unit" not in st.session_state:
            st.session_state.edited_unit = []
        # if 'edited_weight' not in st.session_state:
        #     st.session_state.edited_weight = []
        if "nutrient" not in st.session_state:
            st.session_state.nutrient = {}

        if not st.session_state.generate_value:
            st.session_state.edited_amount = [0] * len(st.session_state.selected_ingres)
            st.session_state.edited_unit = ["g"] * len(st.session_state.selected_ingres)
            # st.session_state.edited_weight = [0] * len(st.session_state.selected_ingres)
            st.session_state.generate_value = True

        data = pd.DataFrame(
            {
                "ingredients": ingre_names,
                "standard_exp": ingre_exps,
                "amount": st.session_state.edited_amount,
                "unit": st.session_state.edited_unit,
                # "weight": st.session_state.edited_weight,
            }
        )

        data["index"] = data.index + 1
        data.set_index("index", inplace=True)

        data_df = st.data_editor(
            data,
            column_config={
                "index": st.column_config.Column("index", width=50),
                "ingredients": "食材名",
                "standard_exp": st.column_config.Column("食品名", width=100),
                "amount": st.column_config.NumberColumn(
                    "数量",
                ),
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
                # "weight": "Weight(g)",
            },
            # width = 500,
            hide_index=False,
        )

        if st.button("入力完了", on_click=set_state, args=[4]):
            st.session_state.dataframe_finish = True
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

            predicted_ingre_names = []
            ingre_infos = {}
            for i, row in data_df.iterrows():
                item = st.session_state.selected_ingres[i - 1]
                amount = row["amount"]
                unit = row["unit"]

                ingre_id = ingre_ids[i - 1]
                result = (
                    unit_trans_csv.loc[
                        unit_trans_csv["食品番号"] == int(ingre_id), unit
                    ].iloc[0]
                    if unit != "g"
                    else 1
                )
                result = max(result, 0)
                weight = amount * result

                ingre_infos[ingre_id] = {
                    "amount": amount,
                    "unit": unit,
                    "weight": weight,
                }
                # data_df.at[i, 'weight'] = weight

                st.session_state.edited_amount.append(amount)
                st.session_state.edited_unit.append(unit)
                # st.session_state.edited_weight.append(weight)

                print("idx:", item, "ingreid:", ingre_id, "weight:", weight)
                nutri_list = []
                for code in nutrient_codes:
                    try:
                        nutri_list.append(
                            float(nutrients_infos[ingre_id][code]) * weight / 100
                        )
                    except ValueError:
                        nutri_list.append(float(0))

                new_ing_dic = {ingres[item]: nutri_list}
                nutrient.update(new_ing_dic)
                predicted_ingre_names.append(ingres[item])
            st.session_state.nutrient = nutrient

            if st.session_state.dataframe_finish == True:
                total_df = pd.DataFrame(nutrient)
                total_df["主要栄養素"] = pd.Categorical(
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

                st.bar_chart(
                    total_df, x="主要栄養素", y=predicted_ingre_names
                )  # , color=custom_colors
                st.bar_chart(
                    total_df, x="主要栄養素", y=predicted_ingre_names
                )  # , color=custom_colors
