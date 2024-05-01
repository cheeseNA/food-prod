import json
import os
import random
from datetime import datetime

# import japanese_clip as ja_clip
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

    # # text_probs_np = normalized_tensor.numpy()
    # save_path = '/var/www/html/l_wang/DAI/test_probs.npy'
    # # np.save(save_path, text_probs_np)

    # loaded_np_array = np.load(save_path)
    # normalized_tensor = torch.from_numpy(loaded_np_array)

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


def page_1(username):
    st.title("材料リストによる食事管理")

    if "ingre_finish" not in st.session_state:
        st.session_state.ingre_finish = False

    if "dataframe_finish" not in st.session_state:
        st.session_state.dataframe_finish = False

    if "init_prediction" not in st.session_state:
        st.session_state.init_prediction = False

    if "init_1" not in st.session_state:
        st.session_state.username = True
        # if 'username' not in st.session_state:
        st.session_state.username = username
        # if 'click_dict' not in st.session_state:
        st.session_state.click_dict = {"button": 0, "checkbox": 0, "input_text": 0}
        # if 'stage' not in st.session_state:
        st.session_state.stage = 0
        st.session_state.ingres, st.session_state.name2idx = get_ingres_name()
        # if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

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

    if (
        uploaded_image
        and st.session_state.stage == 0
        and st.session_state.predict_method
    ):
        st.session_state.stage += 1

    # Predict Module
    c1, c2, c3 = st.columns((1, 1, 1))
    # if st.session_state.uploaded_image is not None and st.session_state.stage >= 1:
    if st.session_state.stage >= 1:
        threshold = 0.5
        candidate_nums = 10
        ingres, name2idx = st.session_state.ingres, st.session_state.name2idx
        # display image
        c1.image(st.session_state.uploaded_image, use_column_width=False, width=150)
        image = Image.open(st.session_state.uploaded_image)

        if "init_2" not in st.session_state:
            st.session_state.init_2 = True
            # if 'predict_ingres' not in st.session_state:
            st.session_state.predict_ingres = []
            # if 'selected_ingres' not in st.session_state:
            st.session_state.selected_ingres = []
            # if 'probability_scores' not in st.session_state:
            # text_probs = get_ingre_probability(ingres, image)
            text_probs = get_ingre_prob_from_model(uploaded_image)
            st.session_state.probability_scores = [
                item for sublist in text_probs.tolist() for item in sublist
            ]
            # if 'normalized_matrix' not in st.session_state:
            normalized_matrix = get_normalized_co_occurrence_matrix()
            st.session_state.normalized_matrix = np.where(
                normalized_matrix < threshold / 100, 0, normalized_matrix
            )
            # if 'mask' not in st.session_state:
            mask = [1] * 588
            mask[2] = 0
            mask = np.array(mask)
            st.session_state.mask = mask
            # if 'selected_options' not in st.session_state:
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
                st.session_state.predict_ingres = get_current_candidate(
                    candidate_nums,
                    st.session_state.probability_scores,
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
            item
            for item, selected in st.session_state.selected_options.items()
            if selected
        ]
        # st.session_state.click_dict["checkbox"] += 1
        if options:
            for option in options:
                st.session_state.selected_ingres.append(name2idx[option])
            st.session_state.click_dict["input_text"] += 1

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
            if st.session_state.predict_method == "method_1":
                st.session_state.predict_ingres = get_current_candidate_method1(
                    candidate_nums,
                    st.session_state.probability_scores,
                    st.session_state.mask,
                    st.session_state.normalized_matrix,
                )
            if st.session_state.predict_method == "method_2":
                st.session_state.predict_ingres = get_current_candidate(
                    candidate_nums,
                    st.session_state.probability_scores,
                    st.session_state.mask,
                    st.session_state.normalized_matrix,
                )

            print("generate:", st.session_state.predict_ingres)
            st.rerun()

        c3.divider()
        c3.write("選択された食材リスト:")
        for item in st.session_state.selected_ingres:
            c3.checkbox(ingres[int(item)], value=True)
        c3.divider()

        if c3.button("完了"):
            st.session_state.ingre_finish = True
            set_state(3)

        # if st.session_state.stage >= 3:
        if st.session_state.ingre_finish == True:
            st.session_state.stage = 3
            st.session_state.click_dict["checkbox"] += (
                len(st.session_state.selected_ingres)
                - st.session_state.click_dict["input_text"]
            )
            save_results(
                st.session_state.username,
                image,
                st.session_state.predict_method,
                st.session_state.selected_ingres,
                ingres,
                st.session_state.click_dict,
            )
            c3.success("回答を記録しました！次のページにお進みください。")

            with open("Labels/foodid_dict.json", "r", encoding="utf-8") as file:
                foodid_dic = json.load(
                    file
                )  # {"1": [id, exp, label_name, [whole_exps]]}

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
                st.session_state.edited_amount = [0] * len(
                    st.session_state.selected_ingres
                )
                st.session_state.edited_unit = ["g"] * len(
                    st.session_state.selected_ingres
                )
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
                        "重さ",
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

                # if st.session_state.stage >= 4:
                if st.session_state.dataframe_finish == True:
                    total_df = pd.DataFrame(st.session_state.nutrient)
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


############
usernames = [
    "l_wang",
    "yamakata",
    "admin",
    "f001",
    "f002",
    "f003",
    "f004",
    "f005",
    "f006",
    "f007",
    "f008",
    "f009",
    "f010",
]
passwords = [
    "12345",
    "54321",
    "ad1234",
    "foodlog001",
    "foodlog002",
    "foodlog003",
    "foodlog004",
    "foodlog005",
    "foodlog006",
    "foodlog007",
    "foodlog008",
    "foodlog009",
    "foodlog0010",
]

# ['u001', 'u002', 'u003', 'u004','u005', 'u006','u007', 'u008','u009', 'u010','u011', 'u012','u013', 'u014','u015', 'u016','u017', 'u018','u019', 'u020']
# ['45260', '75340', '44107', '62902', '23960', '75281', '99990', '86609', '76326', '43646', '72447', '75362', '07098', '75855', '26734', '24302', '03279', '86585', '15465', '45127']

credentials = {}
for uname, pwd in zip(usernames, passwords):
    # user_dict = {"name": uname, "password": pwd}
    credentials[uname] = pwd


def authenticate(username, password):
    stored_password = credentials.get(username)

    if password == stored_password:
        return True
    else:
        return False


def main(username):
    if "register" not in st.session_state:
        st.session_state.register = False

    st.set_page_config(
        page_title="RecipeLog2023",
        page_icon=":curry:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if st.session_state.register == False:
        username = st.text_input("アカウント:")
        password = st.text_input("パスワード:", type="password")

        if st.button("登录"):
            if authenticate(username, password):
                st.success("Success! Welcome {}".format(username))
                st.session_state.register = True
                st.rerun()
            else:
                st.error("Username/password is incorrect")

    if st.session_state.register == True:
        methods = ["method_1", "method_2"]
        if "predict_method" not in st.session_state:
            st.session_state.predict_method = random.choice(methods)
            # st.session_state.predict_method = "method_1"

        if "now_page" not in st.session_state:
            st.session_state.now_page = 1

        if st.session_state.now_page == 1:
            page_1(username)


if __name__ == "__main__":
    username = "admin"
    main(username)
