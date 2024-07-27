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
    dish_number = int(request.args.get("dish_number"))
    print(f"{username=}, {meal_time=}, {dish_number=}")
    _, record = get_latest_image_and_record(username, meal_time, dish_number)
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
    dish_number = int(request.args.get("dish_number"))
    image_path, _ = get_latest_image_and_record(username, meal_time, dish_number)
    return send_file(image_path)


def get_latest_image_and_record(
    username: str, meal_time: str, dish_number: int
) -> tuple[str, dict] | None:
    """
    Get the latest image and record of the dish, specified by the arguments.
    If the dish does not exist, return None.
    """
    dish_dir_path = Path(f"records/{username}/{meal_time}/{dish_number}")
    if not dish_dir_path.exists():
        return None
    record_images = [f for f in dish_dir_path.iterdir() if f.suffix == ".jpg"]
    record_images.sort(key=lambda x: datetime.fromisoformat(x.stem))
    latest_image_fullpath = record_images[-1].resolve()
    latest_json = None
    with (dish_dir_path / (record_images[-1].stem + ".json")).open() as f:
        latest_json = json.load(f)
    return latest_image_fullpath, latest_json
