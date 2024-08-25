import json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, send_file

app = Flask(__name__)


@app.route("/")
def result():
    """
    Render the result page.
    """
    FONT_SIZE = 25
    HEIGHT = 500

    username = request.args.get("username")
    meal_time = request.args.get("meal_time")
    dish_number = request.args.get("dish_number", -1, type=int)
    if username is None or meal_time is None or dish_number == -1:
        return "Invalid arguments."
    print(f"{username=}, {meal_time=}, {dish_number=}")
    img_and_record = get_latest_image_and_record(username, meal_time, dish_number)
    if img_and_record is None:
        return "No such dish."
    _, record, main_nutri_fig, pfc_fig, environment_fig = img_and_record
    return render_template(
        "result.html",
        username=username,
        meal_time=datetime.fromisoformat(meal_time).strftime("%Y年%m月%d日 %H時%M分"),
        dish_number=dish_number,
        record=record,
        host_url=request.host_url,
        main_nutri_fig=update_graph_str_for_visibility(
            main_nutri_fig, FONT_SIZE, HEIGHT
        ),
        pfc_fig=update_graph_str_for_visibility(pfc_fig, FONT_SIZE, HEIGHT),
        environment_fig=update_graph_str_for_visibility(
            environment_fig, FONT_SIZE, HEIGHT
        ),
    )


@app.route("/image")
def image():
    """
    Return the image file.
    """
    username = request.args.get("username")
    meal_time = request.args.get("meal_time")
    dish_number = request.args.get("dish_number", -1, type=int)
    if username is None or meal_time is None or dish_number == -1:
        return "Invalid arguments."
    print(f"{username=}, {meal_time=}, {dish_number=}")
    image_and_record = get_latest_image_and_record(username, meal_time, dish_number)
    if image_and_record is None:
        return "No such dish."
    image_path, _, _, _, _ = image_and_record
    return send_file(image_path)


def get_latest_image_and_record(
    username: str, meal_time: str, dish_number: int
) -> tuple[str, dict, str, str, str] | None:
    """
    Get the latest image and record of the dish, specified by the arguments.
    If the dish does not exist, return None.
    """
    dish_dir_path = (
        Path(__file__).parent.parent / f"records/{username}/{meal_time}/{dish_number}"
    )
    if not dish_dir_path.exists():
        return None
    record_images = [f for f in dish_dir_path.iterdir() if f.suffix == ".jpg"]
    record_images.sort(key=lambda x: datetime.fromisoformat(x.stem))
    latest_image_fullpath = record_images[-1].resolve()
    latest_json = None
    main_nutri_fig = None
    pfc_fig = None
    environment_fig = None
    with (dish_dir_path / (record_images[-1].stem + ".json")).open(
        encoding="utf-8"
    ) as f:
        latest_json = json.load(f)
    with (
        dish_dir_path / (record_images[-1].stem + "_main_nutri_fig.json")
    ).open() as f:
        main_nutri_fig = f.read()
    with (dish_dir_path / (record_images[-1].stem + "_pfc_fig.json")).open() as f:
        pfc_fig = f.read()
    with (
        dish_dir_path / (record_images[-1].stem + "_environment_fig.json")
    ).open() as f:
        environment_fig = f.read()
    return latest_image_fullpath, latest_json, main_nutri_fig, pfc_fig, environment_fig


def update_graph_str_for_visibility(fig_json: str, font_size: int, height: int) -> str:
    """
    Update the font size and height of the graph JSON string.
    """
    fig_dict = json.loads(fig_json)
    if "layout" not in fig_dict:
        fig_dict["layout"] = {"font": {"size": font_size}}
    elif "font" not in fig_dict["layout"]:
        fig_dict["layout"]["font"] = {"size": font_size}
    else:
        fig_dict["layout"]["font"]["size"] = font_size
    fig_dict["layout"]["height"] = height
    return json.dumps(fig_dict)
