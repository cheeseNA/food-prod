import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from locales.locale import generate_localer


# jsonファイルを読み込む関数
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def render_record():
    l = generate_localer(st.session_state.lang)
    username = st.session_state.username

    user_dir_path = Path(f"records/{username}")
    if not user_dir_path.exists():
        user_dir_path.mkdir()
    meal_dirs = [d for d in user_dir_path.iterdir() if d.is_dir()]

    st.html(l("あなたの食事記録: ") + username)

    for meal_dir in meal_dirs:
        meal_datetime = meal_dir.name
        st.html(l("食事時刻: ") + meal_datetime)
        dish_dirs = [d for d in meal_dir.iterdir() if d.is_dir()]
        if not dish_dirs:
            st.write(l("料理が登録されていません"))
            continue
        cols = st.columns(len(dish_dirs))
        for i, dish_dir in enumerate(dish_dirs):
            dish_number = dish_dir.name
            cols[i].html(l("料理番号: ") + dish_number)
            record_images = [f for f in dish_dir.iterdir() if f.suffix == ".jpg"]

            # 最新のSONファイルのパスを取得
            record_images.sort(key=lambda x: datetime.fromisoformat(x.stem))
            latest_image = record_images[-1]
            cols[i].image(str(latest_image), width=120)
            if cols[i].button(l("編集"), key=dish_dir.resolve()):
                cols[i].json(read_json(dish_dir / (latest_image.stem + ".json")))
