from flask import Flask
from flask import render_template
from flask import request
import pandas as pd
from flask import send_file
import io

from model_scoring import (
    predict_main_ideas,
    score_student_answer
)

app = Flask(__name__)
RESULT_DF = None

#หน้า home_page
@app.route("/")
def home():
    return render_template("home_page.html")

#หน้า single Answer
@app.route("/single")
def single():
    return render_template(
        "single.html"
    )

# รับคำตอบจาก Single Answer
@app.route("/score_single", methods=["POST"])
def score_single():

    text_301 = request.form.get("answer_301", "").strip()
    text_302 = request.form.get("answer_302", "").strip()

    numline_302 = request.form.get("numline_302", "0")

    # ------------------
    # ข้อ 30.1
    # ------------------

    if text_301:

        result_301 = predict_main_ideas(text_301)

    else:

        result_301 = {
            "S1_SCORE": 0,
            "S2_SCORE": 0,
            "S3_SCORE": 0,
            "S4_SCORE": 0,
            "S5_SCORE": 0,
            "S6_SCORE": 0,

            "S1_REASON": "ไม่มีคำตอบ",
            "S2_REASON": "ไม่มีคำตอบ",
            "S3_REASON": "ไม่มีคำตอบ",
            "S4_REASONS": "ไม่มีคำตอบ",
            "S5_REASON": "ไม่มีคำตอบ",
            "S6_REASON": "ไม่มีคำตอบ",

            "TOTAL_SCORE": 0
        }

    # ------------------
    # ข้อ 30.2
    # ------------------

    if text_302:

        result_302 = score_student_answer(
            text_302,
            numline_302
        )

    else:

        result_302 = {
            "s7_score": 0,
            "s8_score": 0,
            "s9_score": 0,
            "s10_score": 0,
            "s11_score": 0,
            "s12_score": 0,
            "s13_score": 0,

            "s7_info": "ไม่มีคำตอบ",
            "s8_reason": "ไม่มีคำตอบ",
            "s9_reason": "ไม่มีคำตอบ",
            "s10_reason": "ไม่มีคำตอบ",
            "s11_reason": "ไม่มีคำตอบ",
            "s12_reason": "ไม่มีคำตอบ",
            "s13_reason": "ไม่มีคำตอบ",

            "TOTAL_SCORE": 0
        }

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
@app.route("/score_excel", methods=["POST"])
def score_excel():

    file = request.files["excel_file"]

    df = pd.read_excel(file)
    df = df.fillna("ไม่มีคำตอบ")
    df.columns = df.columns.str.strip()

    # ======================
    # เก็บคะแนนทั้งหมด
    # ======================

    score_301_list = []
    score_302_list = []

    s1_list = []
    s2_list = []
    s3_list = []
    s4_list = []
    s5_list = []
    s6_list = []

    s7_list = []
    s8_list = []
    s9_list = []
    s10_list = []
    s11_list = []
    s12_list = []
    s13_list = []

    #เก็บ reason
    s1_reason_list = []
    s2_reason_list = []
    s3_reason_list = []
    s4_reason_list = []
    s5_reason_list = []
    s6_reason_list = []

    s7_info_list = []
    s8_reason_list = []
    s9_reason_list = []
    s10_reason_list = []
    s11_reason_list = []
    s12_reason_list = []
    s13_reason_list = []

    # ======================
    # คำนวณคะแนน
    # ======================

    for _, row in df.iterrows():

        text_301 = str(row["TEXT_301"]).strip()
        text_302 = str(row["TEXT_302"]).strip()
        numline_302 = int(row["NUMLINE_302"])

        # -------- 30.1 --------
        result_301 = predict_main_ideas(text_301)
        print(result_301)

        s1_reason_list.append(result_301.get("S1_REASON", ""))
        s2_reason_list.append(result_301.get("S2_REASON", ""))
        s3_reason_list.append(result_301.get("S3_REASON", ""))
        s4_reason_list.append(result_301.get("S4_REASONS", ""))
        s5_reason_list.append(result_301.get("S5_REASON", ""))
        s6_reason_list.append(result_301.get("S6_REASON", ""))

        s1_list.append(result_301.get("S1_SCORE", 0))
        s2_list.append(result_301.get("S2_SCORE", 0))
        s3_list.append(result_301.get("S3_SCORE", 0))
        s4_list.append(result_301.get("S4_SCORE", 0))
        s5_list.append(result_301.get("S5_SCORE", 0))
        s6_list.append(result_301.get("S6_SCORE", 0))

        score_301_list.append(
            result_301.get("TOTAL_SCORE", 0)
        )

        # -------- 30.2 --------
        result_302 = score_student_answer(
            text_302,
            numline_302
        )

        s7_info_list.append(result_302.get("s7_info", ""))

        s8_reason_list.append(result_302.get("s8_reason", ""))
        s9_reason_list.append(result_302.get("s9_reason", ""))

        s10_reason_list.append(result_302.get("s10_reason", ""))
        s11_reason_list.append(result_302.get("s11_reason", ""))

        s12_reason_list.append(result_302.get("s12_reason", ""))
        s13_reason_list.append(result_302.get("s13_reason", ""))

        s7_list.append(result_302.get("s7_score", 0))
        s8_list.append(result_302.get("s8_score", 0))
        s9_list.append(result_302.get("s9_score", 0))
        s10_list.append(result_302.get("s10_score", 0))
        s11_list.append(result_302.get("s11_score", 0))
        s12_list.append(result_302.get("s12_score", 0))
        s13_list.append(result_302.get("s13_score", 0))

        score_302_list.append(
            result_302.get("TOTAL_SCORE", 0)
        )

    # ======================
    # เพิ่มคอลัมน์ลง df
    # ======================

    df["SCORE_301"] = score_301_list
    df["SCORE_302"] = score_302_list

    df["S1_SCORE"] = s1_list
    df["S2_SCORE"] = s2_list
    df["S3_SCORE"] = s3_list
    df["S4_SCORE"] = s4_list
    df["S5_SCORE"] = s5_list
    df["S6_SCORE"] = s6_list

    df["S7_SCORE"] = s7_list
    df["S8_SCORE"] = s8_list
    df["S9_SCORE"] = s9_list
    df["S10_SCORE"] = s10_list
    df["S11_SCORE"] = s11_list
    df["S12_SCORE"] = s12_list
    df["S13_SCORE"] = s13_list

    df["S1_REASON"] = s1_reason_list
    df["S2_REASON"] = s2_reason_list
    df["S3_REASON"] = s3_reason_list
    df["S4_REASON"] = s4_reason_list
    df["S5_REASON"] = s5_reason_list
    df["S6_REASON"] = s6_reason_list

    df["S7_INFO"] = s7_info_list
    df["S8_REASON"] = s8_reason_list
    df["S9_REASON"] = s9_reason_list
    df["S10_REASON"] = s10_reason_list
    df["S11_REASON"] = s11_reason_list
    df["S12_REASON"] = s12_reason_list
    df["S13_REASON"] = s13_reason_list

    df["TOTAL_SCORE"] = (
        df["SCORE_301"] +
        df["SCORE_302"]
    )

    global RESULT_DF
    RESULT_DF = df.copy()

    # ======================
    # สร้าง HTML
    # ======================

    rows_html = ""

    for _, row in df.iterrows():

        rows_html += f"""
        <tr>

            <td>{row['PAPER_CODE']}</td>

            <td>
                <details>
                    <summary>ดูคำตอบ</summary>
                    {row['TEXT_301']}
                </details>
            </td>

            <td>
                <details>
                    <summary>ดูคำตอบ</summary>
                    {row['TEXT_302']}
                </details>
            </td>

            <td>{row['NUMLINE_302']}</td>

            <td>
                <details>
                    <summary>{row['SCORE_301']}/10</summary>

                    <b>S1</b> = {row['S1_SCORE']}<br>
                    {row['S1_REASON']}<br><br>

                    <b>S2</b> = {row['S2_SCORE']}<br>
                    {row['S2_REASON']}<br><br>

                    <b>S3</b> = {row['S3_SCORE']}<br>
                    {row['S3_REASON']}<br><br>

                    <b>S4</b> = {row['S4_SCORE']}<br>
                    {row['S4_REASON']}<br><br>

                    <b>S5</b> = {row['S5_SCORE']}<br>
                    {row['S5_REASON']}<br><br>

                    <b>S6</b> = {row['S6_SCORE']}<br>
                    {row['S6_REASON']}
                </details>
            </td>

            <td>
                <details>
                    <summary>{row['SCORE_302']}/20</summary>

                    <b>S7</b> = {row['S7_SCORE']}<br>
                    {row['S7_INFO']}<br><br>

                    <b>S8</b> = {row['S8_SCORE']}<br>
                    {row['S8_REASON']}<br><br>

                    <b>S9</b> = {row['S9_SCORE']}<br>
                    {row['S9_REASON']}<br><br>

                    <b>S10</b> = {row['S10_SCORE']}<br>
                    {row['S10_REASON']}<br><br>

                    <b>S11</b> = {row['S11_SCORE']}<br>
                    {row['S11_REASON']}<br><br>

                    <b>S12</b> = {row['S12_SCORE']}<br>
                    {row['S12_REASON']}<br><br>

                    <b>S13</b> = {row['S13_SCORE']}<br>
                    {row['S13_REASON']}
                </details>
            </td>

            <td>{row['TOTAL_SCORE']}</td>

        </tr>
        """

    return render_template(
        "excel_score.html",
        rows_html=rows_html
    )

#ดาวน์โหลด Excel ที่มีผลการตรวจ
@app.route("/download_excel")
def download_excel():

    global RESULT_DF

    if RESULT_DF is None:
        return "ยังไม่มีผลการตรวจ กรุณาอัปโหลดไฟล์ก่อน"

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # ==================================
        # SHEET 1 : คะแนนรวม
        # ==================================

        summary_cols = [
            "PAPER_CODE",
            "SCORE_301",
            "SCORE_302",
            "TOTAL_SCORE"
        ]

        RESULT_DF[summary_cols].to_excel(
            writer,
            sheet_name="RESULT",
            index=False
        )

        # ==================================
        # SHEET 2 : คำตอบนักเรียน
        # ==================================

        answer_cols = [
            "PAPER_CODE",
            "TEXT_301",
            "TEXT_302",
            "NUMLINE_302"
        ]

        RESULT_DF[answer_cols].to_excel(
            writer,
            sheet_name="ANSWERS",
            index=False
        )

        # ==================================
        # SHEET 3 : คะแนนย่อย + เหตุผล
        # ==================================

        RESULT_DF.to_excel(
            writer,
            sheet_name="DETAIL",
            index=False
        )

        # ==================================
        # SHEET 4 : เกณฑ์การให้คะแนน
        # ==================================

        criteria_df = pd.DataFrame([
            ["30.1", "S1", "ใจความสำคัญ", 4],
            ["30.1", "S2", "การเรียงลำดับและเชื่อมโยงความคิด", 2],
            ["30.1", "S3", "ความถูกต้องตามหลักการเขียนย่อความ", 1],
            ["30.1", "S4", "การสะกดคำ", 1],
            ["30.1", "S5", "การใช้คำ/ถ้อยคำสำนวน", 1],
            ["30.1", "S6", "การใช้ประโยค", 1],

            ["30.2", "S7", "คำบอกข้อคิดเห็น", 1],
            ["30.2", "S8", "เหตุผลสนับสนุน", 8],
            ["30.2", "S9", "การเรียงลำดับและเชื่อมโยงความคิด", 3],
            ["30.2", "S10", "ความถูกต้องตามหลักการแสดงความคิดเห็น", 2],
            ["30.2", "S11", "การสะกดคำ", 2],
            ["30.2", "S12", "การใช้คำ/ถ้อยคำสำนวน", 2],
            ["30.2", "S13", "การใช้ประโยค", 2],
        ],
        columns=[
            "ข้อ",
            "รหัส",
            "รายละเอียด",
            "คะแนนเต็ม"
        ])

        criteria_df.to_excel(
            writer,
            sheet_name="CRITERIA",
            index=False
        )

        # ==================================
        # ปรับความกว้างคอลัมน์
        # ==================================

        for sheet in writer.book.worksheets:

            for column in sheet.columns:

                max_length = 0

                column_letter = column[0].column_letter

                for cell in column:

                    try:
                        max_length = max(
                            max_length,
                            len(str(cell.value))
                        )
                    except:
                        pass

                adjusted_width = min(
                    max_length + 5,
                    100
                )

                sheet.column_dimensions[
                    column_letter
                ].width = adjusted_width

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="score_result.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

#รันเว็บ
if __name__ == "__main__":
    app.run(
        debug=True
    )