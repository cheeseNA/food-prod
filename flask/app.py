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
    username = request.args.get("username")
    meal_time = request.args.get("meal_time")
    dish_number = request.args.get("dish_number", -1, type=int)
    if username is None or meal_time is None or dish_number == -1:
        return "Invalid arguments."
    print(f"{username=}, {meal_time=}, {dish_number=}")
    img_and_record = get_latest_image_and_record(username, meal_time, dish_number)
    if img_and_record is None:
        return "No such dish."
    _, record = img_and_record
    return render_template(
        "result.html",
        username=username,
        meal_time=meal_time,
        dish_number=dish_number,
        record=record,
        host_url=request.host_url,
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
    image_path, _ = image_and_record
    return send_file(image_path)


def get_latest_image_and_record(
    username: str, meal_time: str, dish_number: int
) -> tuple[str, dict] | None:
    """
    Get the latest image and record of the dish, specified by the arguments.
    If the dish does not exist, return None.
    """
    dish_dir_path = Path(__file__).parent.parent / f"records/{username}/{meal_time}/{dish_number}"
    if not dish_dir_path.exists():
        return None
    record_images = [f for f in dish_dir_path.iterdir() if f.suffix == ".jpg"]
    record_images.sort(key=lambda x: datetime.fromisoformat(x.stem))
    latest_image_fullpath = record_images[-1].resolve()
    latest_json = None
    with (dish_dir_path / (record_images[-1].stem + ".json")).open(encoding="utf-8") as f:
        latest_json = json.load(f)
    return latest_image_fullpath, latest_json
