import json

import streamlit as st

from locales.locale import generate_localer, get_current_lang


def main():
    st.session_state.lang = get_current_lang()
    l = generate_localer(st.session_state.lang)

    st.set_page_config(
        page_title="RecipeLog2023",
        page_icon=":curry:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    users = json.load(open("userdata/users.json"))
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.title("Login")
        c1, _, _ = st.columns((1, 1, 1))
        username = c1.text_input(l("アカウント:"))
        password = c1.text_input(l("パスワード:"), type="password")

        if c1.button("Login"):
            if username in users and users[username]["password"] == password:
                st.success(l("ログイン成功"))
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error(l("ログイン失敗"))
        return

    st.title("User Page")
    c1, _, _ = st.columns((1, 1, 1))
    sex_option = c1.selectbox(
        l("性別"),
        (l("男性"), l("女性")),
        index=0 if users[st.session_state.username]["sex"] == "male" else 1,
    )
    age = c1.number_input(
        l("年齢"),
        min_value=1,
        max_value=100,
        value=users[st.session_state.username]["age"],
    )
    physical_activity_level = c1.selectbox(
        l("身体活動レベル"),
        ("I", "II", "III"),
        index=users[st.session_state.username]["physical_activity_level"] - 1,
    )
    st.write(l("身体活動レベル I: 生活の大部分が座位で、静的な活動が中心の場合"))
    st.write(
        l(
            "身体活動レベル II: 座位中心の仕事だが、職場内での移動や立位での作業・接客等、通勤・買い物での歩行、家事、軽いスポーツ、のいずれかを含む場合"
        )
    )
    st.write(
        l(
            "身体活動レベル III: 移動や立位の多い仕事への従事者、あるいは、スポーツ等余暇における活発な運動習慣を持っている場合"
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


if __name__ == "__main__":
    main()
