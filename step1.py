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
from locales.locale import generate_localer, get_current_lang
from src.dataloader import VireoLoader
from src.model_clip import Recognition

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
    l = generate_localer(st.session_state.lang)
    st.title(l("材料リストによる食事管理"))
    debug_print(st.session_state)

    label_to_id_and_names = get_label_to_id_and_names()
    name_to_label = get_name_to_label(label_to_id_and_names)

    def change_stage_to_image_uploaded():
        st.session_state.stage = StreamlitStep.IMAGE_UPLOADED

    uploaded_image = st.file_uploader(
        l("食事の写真をアップロードしてください"),
        type=["jpg", "jpeg", "png"],
        on_change=change_stage_to_image_uploaded,
    )

    if st.session_state.stage == StreamlitStep.SESSION_WHILE_INIT:
        st.session_state.stage = StreamlitStep.WAIT_FOR_IMAGE
        st.session_state.click_dict = {"button": 0, "checkbox": 0, "input_text": 0}

    if st.session_state.stage <= StreamlitStep.WAIT_FOR_IMAGE:
        return

    c1, c2 = st.columns((1, 1))  # visual statements
    if uploaded_image:
        c1.image(uploaded_image, width=250, use_column_width=False)
        image = Image.open(uploaded_image)

    candidate_nums = 10  # stateless variables

    if (
        st.session_state.stage == StreamlitStep.IMAGE_UPLOADED
    ):  # initializations should be done before next stage
        st.session_state.stage = StreamlitStep.WAIT_FOR_INGREDIENT_SELECTION
        st.session_state.start_time = datetime.now()
        st.session_state.selected_options = []
        st.session_state.mask = np.array([1 if i != 2 else 0 for i in range(588)])
        
    if st.session_state.stage <= StreamlitStep.IMAGE_UPLOADED:
        return

    unique = 0
    # collect all selected ingredients
    selected_ingres = [  # 0-indexed int label
        item for item in st.session_state.selected_options
    ]
    st.session_state.mask = update_mask(
        selected_ingres,
        st.session_state.mask,
    )

    predict_ingres = get_current_candidate(  # TODO: remove current selections
        candidate_nums,
        uploaded_image,
        st.session_state.mask,
    )
    debug_print("predict_ingres", predict_ingres)
    debug_print("selected_options", selected_ingres)

    c2.write(l("食材候補：料理に含まれている材料をチェックしてください"))
    for item in st.session_state.selected_options:
        unique+=1
        c2.checkbox(
            label_to_id_and_names[int(item) + 1][
                "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"
            ],
            value=True,
            on_change=lambda x: st.session_state.selected_options.remove(x),
            args=(item,),
            key=unique
        )

    for item in predict_ingres:
        if item in st.session_state.selected_options:
            continue
        unique+=1
        c2.checkbox(
            label_to_id_and_names[int(item) + 1][
                "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"
            ],
            value=False,
            on_change=lambda x: st.session_state.selected_options.append(x),
            args=(item,),
            key=unique
        )

    # Search box
    unique+=1
    not_in_list_multiselect = c2.multiselect(
        l("リストにない食材を検索:"), [
            it[1]["ja_abbr" if st.session_state.lang == "ja" else "en_abbr"]
            for it in label_to_id_and_names.items()],[],
        key=unique
    )

    if not_in_list_multiselect:  # TODO: use onchange to update selected_options
        for name in not_in_list_multiselect:
             st.session_state.selected_options.append(int(name_to_label[name]) - 1)
        st.session_state.click_dict["input_text"] = len(not_in_list_multiselect)
        st.rerun()
    
    if c2.button(l("新しい食材候補を生成する")):
        st.session_state.click_dict["button"] += 1
        for item in predict_ingres:
            if item not in st.session_state.selected_options:
                st.session_state.mask[item] = 0
        st.rerun()

        
    if c2.button(l("完了")):
        st.session_state.stage = StreamlitStep.AFTER_INGREDIENT_SELECTION_INIT

    if st.session_state.stage <= StreamlitStep.WAIT_FOR_INGREDIENT_SELECTION:
        return

    st.session_state.click_dict["checkbox"] = (
        len(selected_ingres) - st.session_state.click_dict["input_text"]
    )

    if st.session_state.stage == StreamlitStep.AFTER_INGREDIENT_SELECTION_INIT:
        save_results(
            st.session_state.username,
            image,
            "method_2",
            selected_ingres,
            {
                label: id_and_names[
                    "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"
                ]
                for label, id_and_names in label_to_id_and_names.items()
            },
            st.session_state.click_dict,
            st.session_state.start_time,
        )
        st.session_state.stage = StreamlitStep.WAIT_FOR_AMOUNT_INPUT
    c2.success(l("回答を記録しました！次のページに進んでください。"))

    ingre_id_to_weights = get_json_from_file("Labels/weight_median.json")

    jpn_eng_expression_df = pd.read_csv("Labels/foodexpList20240612.csv")
    label_to_names_ids = {}
    for record in jpn_eng_expression_df.to_dict("records"):
        label_to_names_ids[record["ingredient_label"]] = {
            "id": record["foodid"],
            "ja_abbr": record["JPN Abbr"],
            "ja_full": record["JPN full"],
            "en_abbr": record["ENG Abbr"],
            "en_full": record["ENG full"],
        }

    ingre_names = []
    ingre_exps = []
    median_weights = []
    for item in selected_ingres:
        label_id = int(item) + 1
        ingre_id = label_to_names_ids[label_id]["id"]
        if st.session_state.lang == "ja":
            ingre_names.append(label_to_names_ids[label_id]["ja_abbr"])
            ingre_exps.append(label_to_names_ids[label_id]["ja_full"])
        else:
            ingre_names.append(label_to_names_ids[label_id]["en_abbr"])
            ingre_exps.append(label_to_names_ids[label_id]["en_full"])
        median_weights.append(ingre_id_to_weights[str(ingre_id)][1])

    st.write(l("一食分に使った量は何グラムですか？"))
    for ii in range(len(ingre_names)):
        value = round(float(median_weights[ii]), 1)
        min_value = 0.0
        max_value = value*2
        step = 0.1
        if value > 10:
            value = round(value)
            min_value=0
            max_value=value*2
            step = 1
        print(ingre_names[ii], min_value, max_value, value )
        if st.session_state.lang == "ja":
            slidelabel = ingre_names[ii]
        else:
            slidelabel = ingre_names[ii]
        median_weights[ii] = st.slider(slidelabel, min_value, max_value, value, step=step)

        
    data = pd.DataFrame(
        {
            "ingredients": ingre_names,
            "amount": median_weights,
            "unit": ["g"] * len(selected_ingres),
            "standard_exp": ingre_exps,
        }
    )
    data["index"] = data.index + 1
    data.set_index("index", inplace=True)

    data_df = st.data_editor(
        data,
        column_config={
            "ingredients": st.column_config.Column(
                l("食材名"),
                width="medium"
            ),
            "standard_exp": st.column_config.Column(
                l("食品名"),
                width="large"
            ),
            "amount": st.column_config.NumberColumn(
                l("重さ"),
                width="small"
            ),
            "unit": st.column_config.SelectboxColumn(
                l("単位"),
                width="small",
                help="The category of the unit",
                options=[
                    "g",
                    l("無単位"),
                    l("枚"),
                    l("本"),
                    l("個片丁株玉房"),
                    l("杯/カップ"),
                    l("半分"),
                    l("摘/少"),
                    l("小/小匙"),
                    l("中"),
                    l("大/大匙"),
                    l("一掴"),
                    l("袋"),
                    l("箱"),
                    l("缶/カン"),
                    "CC",
                    "cm",
                    l("束"),
                    l("合"),
                ],
                required=True,
            ),
        },
        hide_index=True,
    )

    if st.button(l("完了"), key="amount input done"):
        st.session_state.stage = StreamlitStep.FINISH
    if st.session_state.stage <= StreamlitStep.WAIT_FOR_AMOUNT_INPUT:
        return

    food_label_amount_unit = []
    for i, row in data_df.iterrows():
        label = selected_ingres[i - 1]
        food_label_amount_unit.append(
            {
                "ingre_id": label_to_names_ids[int(label) + 1]["id"],
                "amount": row["amount"],
                "unit": row["unit"],
                "canonical_name": label_to_id_and_names[int(label) + 1][
                    "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"
                ],
            }
        )
    nutrients_df = nutrient_calculate.get_nutri_df_from_food_dict(
        food_label_amount_unit
    )

    necessary_nutrients = nutrient_calculate.calculate_necessary_nutrients(
        users[st.session_state.username]["sex"],
        users[st.session_state.username]["age"],
        users[st.session_state.username]["physical_activity_level"],
    )
    necessary_nutrients_per_meal = {
        key: value / 3 for key, value in necessary_nutrients.items()
    }

    percent_df = get_percent_df(nutrients_df, **necessary_nutrients_per_meal)
    percent_df[l("主要栄養素")] = [
        l("カロリー"),
        l("たんぱく質"),
        l("脂質"),
        l("炭水化物"),
        l("塩分"),
    ]
    percent_fig = px.bar(
        percent_df, x=l("主要栄養素"), y=percent_df.columns[1:].tolist()
    ).update_layout(
        yaxis_title=l("1食の目安量に対する割合 (%)"),
        height=300
    )
    for trace in percent_fig.data:
        raw_series = nutrients_df[trace.name]
        raw_series = raw_series.apply(lambda x: f"{x:.2f}").str.cat(
            ["kcal", "g", "g", "g", "g"], sep=" "
        )
        trace["customdata"] = raw_series
        trace["hovertemplate"] = (
            f"{trace.name}<br>" + "%{customdata}<br>%{y:.2f}%<extra></extra>"
        )
    percent_fig.add_hline(y=100.0, line_color="red", line_dash="dash", line_width=1)
    st.plotly_chart(percent_fig)

    st.write(l("あなたの1食あたりの目標栄養摂取量は"))
    st.write(l("カロリー {:.1f} kcal").format(necessary_nutrients_per_meal["kcal"]))
    st.write(l("たんぱく質 {:.1f} g").format(necessary_nutrients_per_meal["protein"]))
    st.write(l("脂質 {:.1f} g").format(necessary_nutrients_per_meal["fat"]))
    st.write(l("炭水化物 {:.1f} g").format(necessary_nutrients_per_meal["carb"]))
    st.write(l("塩分 {:.2f} g です").format(necessary_nutrients_per_meal["salt"]))


users = json.load(open("userdata/users.json", "r"))


def authenticate(username, password):
    if username not in users:
        return False
    if users[username]["password"] == password:
        return True
    else:
        return False


def main():
    st.session_state.lang = get_current_lang()
    l = generate_localer(st.session_state.lang)
    st.set_page_config(
        page_title="RecipeLog2024",
        page_icon=":curry:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "stage" not in st.session_state:
        st.title("Login")
        c1, _, _ = st.columns((1, 1, 1))
        username = c1.text_input(
            l("アカウント:"),
        )
        password = c1.text_input(l("パスワード:"), type="password")

        if c1.button("Login"):
            if not authenticate(username, password):
                c1.error(l("アカウント／パスワードが正しくありません"))
            else:
                st.session_state.username = username
                st.session_state.register = True
                st.session_state.stage = StreamlitStep.SESSION_WHILE_INIT
                st.rerun()
    else:
        page_1()


if __name__ == "__main__":
    main()
