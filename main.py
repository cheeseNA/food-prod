import json
import datetime
from enum import IntEnum

# import japanese_clip as ja_clip
import numpy as np
import streamlit as st
from PIL import Image

from imageproc import get_current_candidate
from locales.locale import generate_localer
from nutrient_calculate import calculate_necessary_nutrients
from recipelog import (
    get_label_to_id_and_names,
    render_ingredient_selectors,
    render_meal_info_tabs,
    render_weight_input,
    save_results,
    update_mask,
)
from record import render_record
from user_page import user_page
from utils import debug_print


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
    st.markdown(
        """
<style>
    div[data-testid="stVerticalBlock"] {
            gap:0.2rem
    }
    .stSlider [data-testid="stTickBar"] {
        display: none;
    }
    .stSlider label {
        display: block;
        text-align: left;
        height: 0px;
    }
</style>
    """,
        unsafe_allow_html=True,
    )
    l = generate_localer(st.session_state.lang)

    if st.session_state.stage == StreamlitStep.SESSION_WHILE_INIT:
        st.session_state.stage = StreamlitStep.WAIT_FOR_IMAGE
        st.session_state.click_dict = {"button": 0, "checkbox": 0, "input_text": 0}

    label_to_id_and_names = get_label_to_id_and_names()

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

    if st.session_state.stage <= StreamlitStep.WAIT_FOR_IMAGE:
        return

    c1, c2 = st.columns((1, 1))
    if uploaded_image:
        c1.image(uploaded_image, width=250, use_column_width=False)
        image = Image.open(uploaded_image)

    if (
        st.session_state.stage == StreamlitStep.IMAGE_UPLOADED
    ):  # initializations that should be done before next stage
        st.session_state.stage = StreamlitStep.WAIT_FOR_INGREDIENT_SELECTION
        st.session_state.start_time = datetime.datetime.now()
        st.session_state.selected_options = []
        st.session_state.mask = np.array([1 if i != 2 else 0 for i in range(588)])

    if st.session_state.stage <= StreamlitStep.IMAGE_UPLOADED:
        return

    st.session_state.mask = update_mask(
        st.session_state.selected_options,
        st.session_state.mask,
    )

    candidate_nums = 10
    predict_ingres = get_current_candidate(
        candidate_nums,
        uploaded_image,
        st.session_state.mask,
    )

    ########################
    ##### Ingredient Input ######
    ########################
    c2.write(l("材料にチェックを入れて下さい。"))

    st.session_state.selected_options = render_ingredient_selectors(
        c2, "new_log_creation", label_to_id_and_names, predict_ingres
    )

    if c2.button(l("新しい食材候補を生成する")):
        st.session_state.click_dict["button"] += 1
        for item in predict_ingres:
            if item not in st.session_state.selected_options:
                st.session_state.mask[item] = 0
        st.rerun()

    debug_print("predict_ingres", predict_ingres)
    debug_print("selected_options", st.session_state.selected_options)

    if c2.button(l("完了")):
        st.session_state.stage = StreamlitStep.AFTER_INGREDIENT_SELECTION_INIT

    if st.session_state.stage <= StreamlitStep.WAIT_FOR_INGREDIENT_SELECTION:
        return

    st.session_state.click_dict["checkbox"] = (
        len(st.session_state.selected_options)
        - st.session_state.click_dict["input_text"]
    )

    if st.session_state.stage == StreamlitStep.AFTER_INGREDIENT_SELECTION_INIT:
        # currently not used
        st.session_state.stage = StreamlitStep.WAIT_FOR_AMOUNT_INPUT

    ########################
    ##### Wight Input ######
    ########################

    weights = render_weight_input(
        st, label_to_id_and_names, st.session_state.selected_options
    )

    food_label_amount_unit = []
    for i, label in enumerate(st.session_state.selected_options):
        food_label_amount_unit.append(
            {
                "ingre_id": label_to_id_and_names[int(label) + 1]["id"],
                "amount": weights[i],
                "unit": "g",
                "canonical_name": label_to_id_and_names[int(label) + 1][
                    "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"
                ][:10],
            }
        )

    necessary_nutrients = calculate_necessary_nutrients(
        st.session_state.users[st.session_state.username]["sex"],
        st.session_state.users[st.session_state.username]["age"],
        st.session_state.users[st.session_state.username]["physical_activity_level"],
    )
    necessary_nutrients_per_meal = {
        key: value / 3 for key, value in necessary_nutrients.items()
    }

    render_meal_info_tabs(food_label_amount_unit, necessary_nutrients_per_meal)

    st.html(  # rethink where to put
        "<b>"
        + str(l("あなたの1食あたりの目標栄養摂取量は"))
        + "</b><br>\n"
        + str(l("カロリー {:.1f} kcal").format(necessary_nutrients_per_meal["kcal"]))
        + "<br>\n"
        + str(l("たんぱく質 {:.1f} g").format(necessary_nutrients_per_meal["protein"]))
        + "<br>\n"
        + str(l("脂質 {:.1f} g").format(necessary_nutrients_per_meal["fat"]))
        + "<br>\n"
        + str(l("炭水化物 {:.1f} g").format(necessary_nutrients_per_meal["carb"]))
        + "<br>\n"
        + str(l("塩分 {:.2f} g です").format(necessary_nutrients_per_meal["salt"]))
    )

    # ユーザに日付を入力させる
    date_input = st.date_input(l("日付を選択してください"))
    # ユーザに時刻を入力させる
    time_input = st.time_input(l("時刻を選択してください"), datetime.time(12, 0))

    if st.button(l("保存"), key="amount input done") and date_input and time_input:
        st.session_state.stage = StreamlitStep.FINISH
    if st.session_state.stage <= StreamlitStep.WAIT_FOR_AMOUNT_INPUT:
        return

    if st.session_state.stage == StreamlitStep.FINISH:
        ingre_names = []
        ingre_ids = []
        for item in st.session_state.selected_options:
            label_id = int(item) + 1
            ingre_id = label_to_id_and_names[label_id]["id"]
            ingre_ids.append(ingre_id)
            ingre_names.append(
                label_to_id_and_names[label_id][
                    "ja_abbr" if st.session_state.lang == "ja" else "en_abbr"
                ]
            )
        save_results(
            st.session_state.username,
            image,
            "default",
            ingre_ids,
            ingre_names,
            weights,
            date_input,
            time_input,
            st.session_state.click_dict,
            st.session_state.start_time,
        )
        st.success(l("食事記録を保存しました。"))
        st.session_state.stage = StreamlitStep.WAIT_FOR_AMOUNT_INPUT


def main():
    st.session_state.users = json.load(
        open("userdata/users.json", "r")
    )  # call at every run to get latest data
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
            "Account / アカウント",
        )
        password = c1.text_input("Password / パスワード", type="password")

        if c1.button("Login"):
            if (
                username not in st.session_state.users
                or st.session_state.users[username]["password"] != password
            ):
                c1.error("Password is incorrect. パスワードが正しくありません。")
            else:
                st.session_state.username = username
                st.session_state.lang = st.session_state.users[username]["lang"]
                st.session_state.register = True
                st.session_state.stage = StreamlitStep.SESSION_WHILE_INIT
                st.rerun()
    else:
        l = generate_localer(st.session_state.lang)
        tab1, tab2, tab3 = st.tabs([l("メイン"), l("ユーザ情報"), l("食事記録")])
        with tab1:
            page_1()
        with tab2:
            user_page()
        with tab3:
            render_record()


if __name__ == "__main__":
    main()
