import json

import streamlit as st

from locales.locale import generate_localer
from nutrient_calculate import calculate_necessary_nutrients


def user_page():
    """
    Used in main.py to display user page
    lang, username should be in session_state before calling this function.
    To avoid impacting other pages, we should not set session_state variables in this function.
    """
    l = generate_localer(st.session_state.lang)
    users = json.load(open("userdata/users.json", "r", encoding="utf-8"))

    current_lang = users[st.session_state.username]["lang"]
    lang_option = st.selectbox(
        l("言語"),
        (l("英語"), l("日本語")),
        index=0 if current_lang == "en" else 1,
    )

    current_sex = users[st.session_state.username]["sex"]
    sex_option = st.selectbox(
        l("性別"),
        (l("男性"), l("女性")),
        index=0 if current_sex == "male" else 1,
    )
    age = st.slider(
        l("年齢"),
        min_value=1,
        max_value=100,
        value=users[st.session_state.username]["age"],
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
    physical_level = ["I", "II", "III"]
    physical_activity_level = st.select_slider(
        l("身体活動レベル"),
        options=physical_level,
        value=physical_level[
            users[st.session_state.username]["physical_activity_level"] - 1
        ],
    )
    if st.button(l("更新")):
        users[st.session_state.username]["lang"] = (
            "en" if lang_option == l("英語") else "ja"
        )
        st.session_state.lang = "en" if lang_option == l("英語") else "ja"
        users[st.session_state.username]["sex"] = (
            "male" if sex_option == l("男性") else "female"
        )
        users[st.session_state.username]["age"] = age
        users[st.session_state.username]["physical_activity_level"] = (
            1
            if physical_activity_level == "I"
            else 2 if physical_activity_level == "II" else 3
        )
        json.dump(users, open("userdata/users.json", "w", encoding="utf-8"), indent=4)
        st.rerun()

    necessary_nutrients = calculate_necessary_nutrients(
        users[st.session_state.username]["sex"],
        users[st.session_state.username]["age"],
        users[st.session_state.username]["physical_activity_level"],
    )
    necessary_nutrients_per_meal = {
        key: value / 3 for key, value in necessary_nutrients.items()
    }
    st.html(
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
