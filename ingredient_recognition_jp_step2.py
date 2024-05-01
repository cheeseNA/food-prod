import os
import io
import requests
from PIL import Image
# import japanese_clip as ja_clip
import numpy as np
import pandas as pd
import streamlit as st
import json
import random

import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename

from src.dataloader import VireoLoader
from src.model_clip import Recognition

device = 'cpu'

def get_ingres_name():
    file_path = 'Labels/IngredientList588.txt'
    ingres = []
    name2idx = {}
    i = 0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.rstrip('\n')
            ingres.append(line)
            name2idx[line] = i
            i += 1
    return ingres, name2idx

def get_normalized_co_occurrence_matrix():
    co_occurrence_matrix = np.load("Labels/co_occurrence_matrix.npy", allow_pickle=True)
    row_sums = co_occurrence_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = co_occurrence_matrix / row_sums
    return normalized_matrix

def get_ingre_probability(ingres, uploaded_image):
    # Load CLIP model
    device = 'cpu'
    model, preprocess = ja_clip.load("rinna/japanese-cloob-vit-b-16", device=device)
    tokenizer = ja_clip.load_tokenizer()

    input_image = preprocess(uploaded_image).unsqueeze(0).to(device)
    search = model.get_image_features(input_image).cpu()

    encodings = ja_clip.tokenize(
    texts=[f"{ing}を使った料理" for ing in ingres],
    max_seq_len=77,
    device=device,
    tokenizer=tokenizer, # this is optional. if you don't pass, load tokenizer each time
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

def get_ingre_prob_from_model(uploaded_image):
    # model_path = '/home/l_wang/vireofood251/Compare_subdatasets/checkpoints_region/Model/clip_vit32/n_epoch21_20.938.pth'
    model_path = 'Models/clip_v32_epoch21_20.938.pth'

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

    criterions = [nn.CrossEntropyLoss().to("cpu"),nn.BCELoss(reduction='none').to("cpu")]

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        VireoLoader(uploaded_image,
            transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize])),
		batch_size = 1, shuffle=True,
		timeout=1000, num_workers=1, 
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
    return normalized_tensor

def update_mask(selected_items, mask, normalized_matrix):
    for selected in selected_items:
        distances = normalized_matrix[selected]
        dist_mask= np.where(distances == 0, 0, 1)
        mask = dist_mask & mask
    return mask

def update_mask_method1(selected_items, mask):
    for selected in selected_items:
        mask[selected] = 0
    return mask

def get_current_candidate(candidate_nums, flat_list, mask, normalized_matrix):
    cur_prob = flat_list * mask
    top_k_indices = np.argsort(cur_prob)[-candidate_nums:][::-1]
    return top_k_indices.tolist()

def get_current_candidate_method1(candidate_nums, flat_list, mask, normalized_matrix):
    cur_prob = flat_list * mask
    top_k_indices = np.argsort(cur_prob)[-candidate_nums:][::-1]
    return top_k_indices.tolist()
    
def save_results(username, image_file, method, ingredients, ingres_convert, click_dict):
    directory = f"Results/{username}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    existing_files = [file for file in os.listdir(directory) if file.startswith("result") and file.endswith(".json")]
    next_serial_number = len(existing_files) + 1

    output_path = f"{directory}test_{next_serial_number}.json"

    ingre_names = []
    for item in ingredients:
        ingre_names.append(ingres_convert[int(item)])

    filename =f"image_{next_serial_number}.png"
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
        "click_dict": click_dict
    }

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(result_data, file, ensure_ascii=False, indent=4)

def set_state(i):
    st.session_state.stage = i
    st.session_state.click_dict["button"] += 1

############
def main(username, uploaded_image, predict_method):
    if 'click_dict' not in st.session_state:
        st.session_state.click_dict = {"button": 0, "checkbox": 0, "input_text": 0}
    click_dict = st.session_state.click_dict

    options = ["方法一", "方法二"]
    if 'predict_method' not in st.session_state:
        if predict_method == options[0]:
            predict_method = options[1]
        else:
            predict_method = options[0]
        st.session_state.predict_method = predict_method
    
    if 'stage' not in st.session_state:
        st.session_state.stage = 0

    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = uploaded_image
    uploaded_image = st.session_state.uploaded_image

    # st.set_page_config(layout="wide") # can only be called once
    st.title("材料リストによる食事管理")

    # st.session_state.predict_method = st.radio("食材候補予測方法を選んでください", ["方法一", "方法二"], index=None)
    if st.session_state.stage == 0 and st.session_state.predict_method:
        st.session_state.stage += 1

    if st.session_state.stage >= 1:
        threshold = 0.5
        candidate_nums = 10

        # uploaded_image = st.file_uploader("食事の写真をアップロードしてください", type=["jpg", "jpeg", "png"])
        # st.session_state.uploaded_image = uploaded_image
        ingres, name2idx = get_ingres_name()

    # Predict Module
    c1, c2, c3 = st.columns((1, 1, 1))
    if uploaded_image is not None and st.session_state.stage >= 1:

        # display image
        # c1.image(uploaded_image, use_column_width=False, width=150)
        image = Image.open(uploaded_image)
        c1.write(st.session_state.predict_method)
        
        if 'predict_ingres' not in st.session_state:
            st.session_state.predict_ingres = []
        predictions = st.session_state.predict_ingres

        if 'selected_ingres' not in st.session_state:
            st.session_state.selected_ingres = []
        selected_ingres = st.session_state.selected_ingres

        if 'probability_scores' not in st.session_state:
            # text_probs = get_ingre_probability(ingres, image)
            text_probs = get_ingre_prob_from_model(uploaded_image)
            st.session_state.probability_scores = [item for sublist in text_probs.tolist() for item in sublist]
        probability_scores = st.session_state.probability_scores

        if 'normalized_matrix' not in st.session_state:
            st.session_state.normalized_matrix = get_normalized_co_occurrence_matrix()
        normalized_matrix = st.session_state.normalized_matrix
        normalized_matrix = np.where(normalized_matrix < threshold/100, 0, normalized_matrix)

        if 'mask' not in st.session_state:
            mask = [1] * 588
            mask = np.array(mask)
            st.session_state.mask = mask
        mask = st.session_state.mask

        if 'selected_options' not in st.session_state:
            st.session_state.selected_options = {}
        selected_options = st.session_state.selected_options

        if st.session_state.stage == 1:
            if st.session_state.predict_method == "方法一":
                predictions = get_current_candidate_method1(candidate_nums, probability_scores, st.session_state.mask, normalized_matrix)
            if st.session_state.predict_method == "方法二":
                predictions = get_current_candidate(candidate_nums, probability_scores, st.session_state.mask, normalized_matrix)
            print(predictions)
            st.session_state.predict_ingres = predictions

            st.session_state.stage += 1

        if st.session_state.stage >= 2:
            c2.write('食材候補：料理に含まれている材料をチェックしてください')
            print("now:",st.session_state.predict_ingres)
            for item in st.session_state.predict_ingres:
                selected_options[item] = c2.checkbox(ingres[int(item)]) # key=f'ingre_{item}')

            options = c2.multiselect(
                'リストにない食材を検索:',
                ingres,
                [])
            selected_ingres = [item for item, selected in selected_options.items() if selected]
            # st.session_state.click_dict["checkbox"] += 1
            if options:
                for option in options:
                    selected_ingres.append(name2idx[option])
                st.session_state.click_dict["input_text"] += 1
            st.session_state.selected_ingres = selected_ingres

            if st.session_state.predict_method == "方法一":
                st.session_state.mask = update_mask_method1(selected_ingres, st.session_state.mask)
            if st.session_state.predict_method == "方法二":
                st.session_state.mask = update_mask(selected_ingres, st.session_state.mask, normalized_matrix)

            if c2.button('新しい食材候補を生成する'):
                st.session_state.click_dict["button"] += 1
                if st.session_state.predict_method == "方法一":
                    st.session_state.predict_ingres = get_current_candidate_method1(candidate_nums, probability_scores, st.session_state.mask, normalized_matrix)
                if st.session_state.predict_method == "方法二":
                    st.session_state.predict_ingres = get_current_candidate(candidate_nums, probability_scores, st.session_state.mask, normalized_matrix)

                print('generate:',st.session_state.predict_ingres)
                st.rerun()
                # print(predictions)
                # st.session_state.predict_ingres = predictions
            
            c3.divider()
            c3.write("選択された食材リスト:")
            # selected_ingres = [43, 31, 8, 56]
            # st.session_state.selected_ingres = selected_ingres
            for item in selected_ingres:
            # st.write(selected_prediction)
        # st.session_state.selected_ingres = options
                c3.checkbox(ingres[int(item)], value=True)
            c3.divider()

            if c3.button('完了', on_click=set_state, args=[3]):
                st.session_state.click_dict["checkbox"] += len(selected_ingres) - st.session_state.click_dict["input_text"]
                save_results(username, image, st.session_state.predict_method, st.session_state.selected_ingres, ingres, st.session_state.click_dict)
                c3.write(st.session_state.click_dict)
            
        if st.session_state.stage >= 3:
            selected_ingres = st.session_state.selected_ingres # ingre_labels
            print(selected_ingres)

            with open('Labels/foodid_dict.json', 'r', encoding='utf-8') as file:
                foodid_dic = json.load(file)  # {"1": [id, exp, label_name, [whole_exps]]}

            ingre_names = []
            ingre_exps = []
            ingre_ids = []
            for item in selected_ingres:
                label_id = int(item)+1
                ingre_names.append(foodid_dic[str(label_id)][2])
                ingre_exps.append(foodid_dic[str(label_id)][1])
                ingre_ids.append(foodid_dic[str(label_id)][0])
            # print(ingre_names)
            # print(ingre_exps)
            # print(ingre_ids)

            if 'generate_value' not in st.session_state:
                st.session_state.generate_value = False
            generate_value = st.session_state.generate_value

            if 'edited_amount' not in st.session_state:
                st.session_state.edited_amount = []
            edited_amount = st.session_state.edited_amount

            if 'edited_unit' not in st.session_state:
                st.session_state.edited_unit = []
            edited_unit = st.session_state.edited_unit

            # if 'edited_gram' not in st.session_state:
            #     st.session_state.edited_gram = []
            # edited_gram = st.session_state.edited_gram

            if not st.session_state.generate_value:
                edited_amount = [random.randint(0, 500) for _ in range(len(selected_ingres))]
                edited_unit = ["g"] * len(selected_ingres)
                # edited_gram = edited_amount[:]
                st.session_state.generate_value = True

            data = pd.DataFrame(
                {
                    "ingredients": ingre_names,
                    "standard_exp": ingre_exps,
                    "amount": edited_amount,
                    "unit": edited_unit,
                    # "amount_gram": edited_gram,
                }
            )

            data['index'] = data.index + 1
            data.set_index('index', inplace=True)

            data_df = st.data_editor(
                data,
                column_config={
                    "index": st.column_config.Column(
                        "index", 
                        width=50),
                    "ingredients": "食材名",
                    "standard_exp": st.column_config.Column(
                        "食品名", 
                        width=100),
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
                            "合"
                        ],
                        required=True,
                    ),
                    # "amount_gram": "Amount(g)",
                },
                # width = 500,
                hide_index=False,
            )

            # data_df = pd.DataFrame(data_list)
            edited_amount = data_df["amount"].tolist()
            edited_unit = data_df["unit"].tolist()
            st.session_state.edited_amount = edited_amount
            st.session_state.edited_unit = edited_unit

            ingre_infos = {}
            for i in range(len(edited_unit)):
                ingre_infos[ingre_ids[i]] = {
                    "amount": edited_amount[i],
                    "unit": edited_unit[i]
                }
            # print(ingre_infos)

            # custom_colors = ['#264653', '#2a9d8f', '#e9c46a', '#e76f51']

            with open('Labels/nutrition_dic.json', 'r', encoding='utf-8') as file:
                nutrients_infos = json.load(file)
            unit_trans_csv = pd.read_csv('Labels/weight_trans.csv')

            predicted_ingre_names = []
            data = {
                '主要栄養素': ['カロリー (kcal)', 'たんぱく質 (g)', '脂質 (g)', '炭水化物 (g)', '塩分 (g)'],
                # ENERC_KCAL: Energy (kcal)
                # PROT-: Protein
                # FAT-: Fat
                # CHOAVLM: Carbohydrate
                # WATER: Water
            }

            selected_ingres = st.session_state.selected_ingres
            nutrient_codes = ["ENERC_KCAL", "PROT-", "FAT-", "CHOAVLM", "NACL_EQ"]
            i = 0
            for item in selected_ingres:
                label_id = int(item)+1
                ingre_id = foodid_dic[str(label_id)][0]
                amount, unit = ingre_infos[ingre_ids[i]]['amount'], ingre_infos[ingre_ids[i]]['unit']
                # print(amount, unit)

                result = unit_trans_csv.loc[unit_trans_csv['食品番号'] == int(ingre_id), unit].iloc[0] if unit != 'g' else 1
                result = max(result, 0)

                weight = amount * result
                # st.session_state.edited_gram[i] = weight

                print("idx:",item,"ingreid:", ingre_id, "weight:",weight)
                nutri_list = []
                for code in nutrient_codes:
                    try:
                        nutri_list.append(float(nutrients_infos[ingre_id][code]) * weight/100)
                    except ValueError:
                        nutri_list.append(float(0))
                i += 1
                new_ing_dic = {ingres[item]: nutri_list}
                data.update(new_ing_dic)
                predicted_ingre_names.append(ingres[item])
            # print(data)

            total_df = pd.DataFrame(data)
            total_df['主要栄養素'] = pd.Categorical(total_df['主要栄養素'], categories=['カロリー (kcal)', 'たんぱく質 (g)', '脂質 (g)', '炭水化物 (g)', '塩分 (g)'], ordered=True)

            st.bar_chart(total_df, x='主要栄養素', y=predicted_ingre_names) # , color=custom_colors

if __name__ == '__main__':
    main()

