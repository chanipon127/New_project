from flask import Flask
from flask import render_template
from flask import request
import pandas as pd

from model_scoring import (
    predict_main_ideas,
    score_student_answer
)

app = Flask(__name__)

#หน้า home_page
@app.route("/")
def home():
    return render_template(
        "home_page.html"
    )

#หน้า single Answer
@app.route("/single")
def single():
    return render_template(
        "single.html"
    )

# รับคำตอบจาก Single Answer
@app.route("/score_single", methods=["POST"])
def score_single():

    text_301 = request.form["answer_301"]
    text_302 = request.form["answer_302"]

    # รับจำนวนบรรทัดจากหน้าเว็บ
    numline_302 = request.form.get("numline_302")

    result_301 = predict_main_ideas(text_301)

    result_302 = score_student_answer(
        text_302,
        numline_302
    )

    print("numline_302 =", numline_302)
    print("RESULT_301 =", result_301)
    print("RESULT_302 =", result_302)

    return render_template(
        "view_score.html",
        result_301=result_301,
        result_302=result_302
    )

#หน้า Excel
@app.route("/excel")
def excel():
    return render_template(
        "excel.html"
    )

#รับไฟล์ Excel จากหน้า Excel
@app.route(
    "/score_excel",
    methods=["POST"]
)
def score_excel():
    # รับไฟล์จากฟอร์ม
    file = request.files[
    "excel_file"
]
    # อ่าน Excel
    df = pd.read_excel(file)
    # เก็บผลลัพธ์
    score_301_list = []
    score_302_list = []

    # วนตรวจทีละแถว
    for _, row in df.iterrows():

        text_301 = str(
            row["answer_301"]
        )

        text_302 = str(
            row["answer_302"]
        )
        # ตรวจทีละแถว
    for _, row in df.iterrows():

        text_301 = str(row["answer_301"]).strip()
        text_302 = str(row["answer_302"]).strip()

        # =====================
        # ตรวจข้อ 30.1
        # =====================

        result_301 = predict_main_ideas(
            text_301
        )

        score_301_list.append(
            result_301["TOTAL_SCORE"]
        )

        # =====================
        # ตรวจข้อ 30.2
        # =====================

        result_302 = score_student_answer(
            text_302
        )

        score_302_list.append(
            result_302["TOTAL_SCORE"]
        )

    # เพิ่มคะแนนลง DataFrame

    df["score_301"] = score_301_list
    df["score_302"] = score_302_list

    # คะแนนรวมทั้งสองข้อ

    df["total_score"] = (
        df["score_301"] +
        df["score_302"]
    )

    # บันทึกไฟล์ผลลัพธ์

    output_path = "results/result.xlsx"

    df.to_excel(
        output_path,
        index=False
    )

    # แสดงผลหน้าเว็บ

    return render_template(
        "view_score.html",
        table=df.to_html(
            classes="table table-striped",
            index=False
        )
    )

#รันเว็บ
if __name__ == "__main__":
    app.run(
        debug=True
    )