import json
import os
from datetime import datetime

import streamlit as st
from PIL import Image

from locales.locale import generate_localer


# jsonファイルを読み込む関数
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def convert_timestamp(encoded_timestamp):
    # エンコードされたタイムスタンプをdatetimeオブジェクトに変換
    dt = datetime.strptime(encoded_timestamp, "%Y-%m-%d_%H-%M-%S")
    # 新しいフォーマットの文字列に変換
    formatted_timestamp = dt.strftime("%Y/%m/%d %H:%M")
    return formatted_timestamp


def record():
    l = generate_localer(st.session_state.lang)
    username = st.session_state.username

    directory_path = f"records/{username}/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # record.jsonを読み込む
    record_data = read_json(os.path.join(directory_path, "record.json"))

    # activeがtrueの要素をrecord_timeの順に並べる
    active_records = {k: v for k, v in record_data.items() if v["active"]}
    sorted_records = sorted(
        active_records.items(), key=lambda x: x[1]["record_time"], reverse=True
    )
    print(sorted_records)

    # Streamlitアプリケーションのタイトル
    st.html(l("あなたの食事記録: ") + username)

    # 各JSONファイルから画像を表示
    for json_file in sorted_records:
        json_path = os.path.join(directory_path, f"record_{json_file[0]}.json")
        data = read_json(json_path)

        timestamp = convert_timestamp(json_file[1]["record_time"])
        st.html(timestamp)
        if "image" in data and "filename" in data["image"] and "path" in data["image"]:
            image_path = data["image"]["path"]
            image = Image.open(image_path)
            st.image(image, width=120)

        # 編集ボタンをクリックしたときにJSON内容を表示する
        if st.button(l("編集"), key=json_file):
            st.json(data)
