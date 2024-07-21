import json


def get_current_lang():
    settings = json.load(open("locales/locale_settings.json", "r", encoding="utf-8"))
    return settings["lang"]


def generate_localer(lang: str):
    ja_to_en = json.load(open("locales/ja_to_en.json", "r", encoding="utf-8"))

    def localer(key: str):
        if lang == "ja" or key not in ja_to_en:
            return key
        return ja_to_en[key]

    return localer
