import json
from pathlib import Path

import streamlit as st

from locales.locale import generate_localer


@st.cache_data
def get_result_from_username_and_uuid(username: str, uuid: str) -> dict | None:
    """
    ユーザー名とuuidから結果を取得する.
    見つからない場合はNoneを返す.
    """
    user_folder_path = (
        Path("Results") / username
    )  # 実行ディレクトリからのパスであることに注意
    if not user_folder_path.exists():
        return None
    for file in user_folder_path.glob("result_*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("uuid") == uuid:
                return data
    return None


def result_page():
    """
    結果共有ページ. ログインなしでもアクセス可能.
    """
    l = generate_localer("en")  # TODO: ユーザーのブラウザの言語設定を利用するようにする
    username = st.query_params.get("user")
    result_uuid = st.query_params.get("id")
    if username is None or result_uuid is None:
        st.error(l("不正なURLです."))
        return
    result = get_result_from_username_and_uuid(username, result_uuid)
    if result is None:
        st.error(l("結果が見つかりません."))
        return
    st.write(result)


if __name__ == "__main__":
    result_page()
