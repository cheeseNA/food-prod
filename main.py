import json
import os
from datetime import datetime
from enum import IntEnum

# import japanese_clip as ja_clip
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from plotly import express as px

from nutrient_calculate import *
from locales.locale import generate_localer, get_current_lang

from recipelog import *
from imageproc import *

import math

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
    #st.title(l("材料リストによる食事管理"))
    debug_print(st.session_state)

    label_to_id_and_names = get_label_to_id_and_names()
    name_to_label = get_name_to_label(label_to_id_and_names)
    nutrition_fact = getNutrientFact()

    ########################
    ##### Image upload  ####
    ########################
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

    ########################
    ##### Ingredient Input ######
    ########################
    c2.write(l("材料にチェックを入れて下さい。"))

    st.markdown("""
<style>
        div[data-testid="stVerticalBlock"] {
            gap:0rem
    }
</style>
    """, unsafe_allow_html=True)
    
    for item in st.session_state.selected_options:
        #print('food:', label_to_id_and_names[int(item) + 1]) 
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
            item[1]["ja_abbr" if st.session_state.lang == "ja" else "en_abbr"]
            for item in label_to_id_and_names.items()],[],
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

    debug_print("predict_ingres", predict_ingres)
    debug_print("selected_ingres", selected_ingres)
    debug_print("selected_options", st.session_state.selected_options)
        
    if c2.button(l("完了")):
        st.session_state.stage = StreamlitStep.AFTER_INGREDIENT_SELECTION_INIT

    if st.session_state.stage <= StreamlitStep.WAIT_FOR_INGREDIENT_SELECTION:
        return

    st.session_state.click_dict["checkbox"] = (
        len(selected_ingres) - st.session_state.click_dict["input_text"]
    )


    ########################
    ##### Wight Input ######
    ########################
    ingre_id_to_weights = get_json_from_file("Labels/weight_median20240615.json")

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
    ingre_ids = []
    for item in selected_ingres:
        label_id = int(item) + 1
        ingre_id = label_to_names_ids[label_id]["id"]
        ingre_ids.append(ingre_id)
        if st.session_state.lang == "ja":
            ingre_names.append(label_to_names_ids[label_id]["ja_abbr"])
            ingre_exps.append(label_to_names_ids[label_id]["ja_full"])
        else:
            ingre_names.append(label_to_names_ids[label_id]["en_abbr"])
            ingre_exps.append(label_to_names_ids[label_id]["en_full"])
        median_weights.append(ingre_id_to_weights[str(ingre_id)][2])

    st.write(l("一食分に使った量は何グラムですか？"))

    with st.container(height=200):
        for ii in range(len(ingre_names)):
            value = round(float(median_weights[ii]), 1)
            min_value = 0.0
            max_value = value*2
            step = 0.1
            if value > 10:
                value = round(value)
                max_value = int(max_value)
                min_value=  int(0)
                step = 1
            else:
                value = float(value)
                min_value = 0.0
                step = 0.1
            
            slidelabel = ingre_names[ii] if len(ingre_names[ii])<40 else ingre_names[ii][:40]
            median_weights[ii] = st.slider(slidelabel, min_value, max_value, value, step=step)

    css = """
<style>
    .stSlider [data-testid="stTickBar"] {
        display: none;
    }
    .stSlider label {
        display: block;
        text-align: left;
        height: 0px;
    }
</style>
"""

    st.markdown(css, unsafe_allow_html=True)
        
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

    food_label_amount_unit = []
    for i, row in data.iterrows():
        label = selected_ingres[i - 1]
        longname = label_to_id_and_names[int(label) + 1][
            "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"]
        shortname = longname if len(longname)<10 else longname[:10]
        food_label_amount_unit.append(
            {
                "ingre_id": label_to_names_ids[int(label) + 1]["id"],
                "amount": row["amount"],
                "unit": row["unit"],
                "canonical_name": shortname,
            }
        )
    nutrients_df = get_nutri_df_from_food_dict(
        food_label_amount_unit
    )

    necessary_nutrients = calculate_necessary_nutrients(
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

    pfc_df = calc_pfc(nutrients_df)
    #print('pfc\n', pfc_df.tolist())
    percent_fig2 = px.pie(
        values=pfc_df.tolist(),
        names=["Protain", "Fat", "Carb"],
        height=350,
    )
        
        
    data_df = nutrition_fact.copy()
    data_df = data_df.loc[ingre_ids]
    data_df = data_df.drop(['index', 'JName'], axis=1)
    data_df.insert(1, 'weight', 0)

    for ii in range(len(data_df)):
        data_df.iloc[ii, 1] = median_weights[ii]
        for jj in range(2, len(data_df.columns)):
            if not math.isnan(data_df.iloc[ii,jj]):
                data_df.iloc[ii,jj] = float(data_df.iloc[ii,jj]) * median_weights[ii] / 100
    append_sum_row_label(data_df)
    #print(data_df)

    tab3, tab4, tab5 = st.tabs([l("主要栄養素"), l("PFCバランス"), l("栄養成分表")])
    with tab3:
        st.plotly_chart(percent_fig, use_container_width=True)
    with tab4:
        st.plotly_chart(percent_fig2, use_container_width=True)
    with tab5:
        st.dataframe(data_df, width=800)

    if st.button(l("保存"), key="amount input done"):
        st.session_state.stage = StreamlitStep.FINISH
    if st.session_state.stage <= StreamlitStep.WAIT_FOR_AMOUNT_INPUT:
        return

    if st.session_state.stage == StreamlitStep.FINISH:
        save_results(
            st.session_state.username,
            image,
            "method_2",
            ingre_ids,
            ingre_names,
            median_weights,
            st.session_state.click_dict,
            st.session_state.start_time,
        )
        st.session_state.stage = StreamlitStep.WAIT_FOR_AMOUNT_INPUT
    c2.success(l("食事記録を保存しました。"))

    necessary_nutrients = calculate_necessary_nutrients(
        users[st.session_state.username]["sex"],
        users[st.session_state.username]["age"],
        users[st.session_state.username]["physical_activity_level"],
    )
    necessary_nutrients_per_meal = {
        key: value / 3 for key, value in necessary_nutrients.items()
    }
    output = '<b>' + str(l("あなたの1食あたりの目標栄養摂取量は")) + '</b><br>\n'\
        + str(l("カロリー {:.1f} kcal").format(necessary_nutrients_per_meal["kcal"])) + '<br>\n'\
        +str(l("たんぱく質 {:.1f} g").format(necessary_nutrients_per_meal["protein"])) + '<br>\n'\
        +str(l("脂質 {:.1f} g").format(necessary_nutrients_per_meal["fat"])) + '<br>\n'\
        +str(l("炭水化物 {:.1f} g").format(necessary_nutrients_per_meal["carb"])) + '<br>\n'\
        +str(l("塩分 {:.2f} g です").format(necessary_nutrients_per_meal["salt"]))
    st.html(output)


users = json.load(open("userdata/users.json", "r"))


def authenticate(username, password):
    if username not in users:
        return False
    if users[username]["password"] == password:
        return True
    else:
        return False


def user_page():
    l = generate_localer(st.session_state.lang)
    users = json.load(open("userdata/users.json"))
    current_sex = users[st.session_state.username]["sex"]

    #st.title("User Page")
    sex_option = st.selectbox(
        l("性別"),
        (l("男性"), l("女性")),
        index=0 if current_sex == "male" else 1,
    )
    age = st.slider(l("年齢"), min_value=1,
        max_value=100,
        value=users[st.session_state.username]["age"],
    )
    phsical_label = ["I", "II", "III"]
    physical_activity_level = st.select_slider(
        l("身体活動レベル"),
        options=phsical_label,
        value=phsical_label[users[st.session_state.username]["physical_activity_level"] - 1],
    )
    st.html(l("<b>レベル I</b>:<br> 生活の大部分が座位で、静的な活動が中心の場合"))
    st.html(
        l(
            "<b>レベル II</b>:<br> 座位中心の仕事だが、職場内での移動や立位での作業・接客等、通勤・買い物での歩行、家事、軽いスポーツ、のいずれかを含む場合"
        )
    )
    st.html(
        l(
            "<b>レベル III</b>:<br> 移動や立位の多い仕事への従事者、あるいは、スポーツ等余暇における活発な運動習慣を持っている場合"
        )
    )
    if st.button(l("更新")):
        users[st.session_state.username]["sex"] = (
            "male" if sex_option == l("男性") else "female"
        )
        users[st.session_state.username]["age"] = age
        users[st.session_state.username]["physical_activity_level"] = (
            1
            if physical_activity_level == "I"
            else 2 if physical_activity_level == "II" else 3
        )
        json.dump(users, open("userdata/users.json", "w"), indent=4)
    
    necessary_nutrients = calculate_necessary_nutrients(
        users[st.session_state.username]["sex"],
        users[st.session_state.username]["age"],
        users[st.session_state.username]["physical_activity_level"],
    )
    necessary_nutrients_per_meal = {
        key: value / 3 for key, value in necessary_nutrients.items()
    }
    output = '<b>' + str(l("あなたの1食あたりの目標栄養摂取量は")) + '</b><br>\n'\
        + str(l("カロリー {:.1f} kcal").format(necessary_nutrients_per_meal["kcal"])) + '<br>\n'\
        +str(l("たんぱく質 {:.1f} g").format(necessary_nutrients_per_meal["protein"])) + '<br>\n'\
        +str(l("脂質 {:.1f} g").format(necessary_nutrients_per_meal["fat"])) + '<br>\n'\
        +str(l("炭水化物 {:.1f} g").format(necessary_nutrients_per_meal["carb"])) + '<br>\n'\
        +str(l("塩分 {:.2f} g です").format(necessary_nutrients_per_meal["salt"]))
    st.html(output)



def main():
    st.session_state.lang = get_current_lang()
    l = generate_localer(st.session_state.lang)
    st.set_page_config(
        page_title="RecipeLog2024",
        page_icon=":curry:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    #st.title("RecipeLog Web")
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
        tab1, tab2 = st.tabs([l("メイン"), l("ユーザ情報")])
        with tab1:
            page_1()

        with tab2:
            user_page()

if __name__ == "__main__":
    main()
