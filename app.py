import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# ==== Load mô hình và preprocessor ====
MODEL_PATH = "xgboost_model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    raise RuntimeError("Không tìm thấy file mô hình hoặc preprocessor. Vui lòng kiểm tra lại.")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ==== Hàm dự đoán ====
def predict_default(
    LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
    PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
    BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
    PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
):
    # Tạo DataFrame
    df_raw = pd.DataFrame([{
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": SEX,
        "EDUCATION": EDUCATION,
        "MARRIAGE": MARRIAGE,
        "AGE": AGE,
        "PAY_1": PAY_1, "PAY_2": PAY_2, "PAY_3": PAY_3,
        "PAY_4": PAY_4, "PAY_5": PAY_5, "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1, "BILL_AMT2": BILL_AMT2, "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4, "BILL_AMT5": BILL_AMT5, "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1, "PAY_AMT2": PAY_AMT2, "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4, "PAY_AMT5": PAY_AMT5, "PAY_AMT6": PAY_AMT6
    }])

    # Feature engineering
    df_raw["no_payment_flag"] = ((df_raw["PAY_1"] == -2) | (df_raw["PAY_2"] == -2) |
                                 (df_raw["PAY_3"] == -2) | (df_raw["PAY_4"] == -2) |
                                 (df_raw["PAY_5"] == -2) | (df_raw["PAY_6"] == -2)).astype(int)

    df_raw["late_but_no_balance_flag"] = (((df_raw["PAY_1"] > 0) & (df_raw["BILL_AMT1"] == 0)) |
                                          ((df_raw["PAY_2"] > 0) & (df_raw["BILL_AMT2"] == 0)) |
                                          ((df_raw["PAY_3"] > 0) & (df_raw["BILL_AMT3"] == 0)) |
                                          ((df_raw["PAY_4"] > 0) & (df_raw["BILL_AMT4"] == 0)) |
                                          ((df_raw["PAY_5"] > 0) & (df_raw["BILL_AMT5"] == 0)) |
                                          ((df_raw["PAY_6"] > 0) & (df_raw["BILL_AMT6"] == 0))).astype(int)

    pay_cols = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df_raw["COUNT_LATE"] = df_raw[pay_cols].apply(lambda row: (row > 0).sum(), axis=1)
    df_raw["MAX_DELAY"] = df_raw[pay_cols].max(axis=1)
    df_raw["LIMIT_BAL_LOG"] = np.log1p(df_raw["LIMIT_BAL"])
    weights = np.array([6,5,4,3,2,1])
    df_raw["RECENCY_WEIGHTED_DELAY"] = (df_raw[pay_cols] * weights).sum(axis=1)
    pay_amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    df_raw["TOTAL_PAYMENT"] = df_raw[pay_amt_cols].sum(axis=1)

    df_raw["overpay_flag"] = (((df_raw["PAY_AMT1"] > df_raw["BILL_AMT1"]) & (df_raw["BILL_AMT1"] > 0)) |
                              ((df_raw["PAY_AMT2"] > df_raw["BILL_AMT2"]) & (df_raw["BILL_AMT2"] > 0)) |
                              ((df_raw["PAY_AMT3"] > df_raw["BILL_AMT3"]) & (df_raw["BILL_AMT3"] > 0)) |
                              ((df_raw["PAY_AMT4"] > df_raw["BILL_AMT4"]) & (df_raw["BILL_AMT4"] > 0)) |
                              ((df_raw["PAY_AMT5"] > df_raw["BILL_AMT5"]) & (df_raw["BILL_AMT5"] > 0)) |
                              ((df_raw["PAY_AMT6"] > df_raw["BILL_AMT6"]) & (df_raw["BILL_AMT6"] > 0))).astype(int)

    bill_amt_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    df_raw["TOTAL_BILL_AMT"] = df_raw[bill_amt_cols].sum(axis=1)
    df_raw["DEBT_TO_LIMIT_RATIO"] = df_raw.apply(
        lambda row: row["TOTAL_BILL_AMT"] / row["LIMIT_BAL"] if row["LIMIT_BAL"] != 0 else 0,
        axis=1
    )

    df_raw = df_raw.drop(columns=["TOTAL_BILL_AMT"])

    # Chọn các feature cần cho preprocessor
    final_features = [
        "no_payment_flag", "late_but_no_balance_flag", "EDUCATION", "COUNT_LATE",
        "MAX_DELAY", "PAY_1", "PAY_2", "TOTAL_PAYMENT", "LIMIT_BAL_LOG",
        "RECENCY_WEIGHTED_DELAY", "SEX", "AGE", "MARRIAGE", "overpay_flag",
        "DEBT_TO_LIMIT_RATIO", "BILL_AMT5"
    ]
    X = df_raw[final_features]

    # Transform + predict
    X_processed = preprocessor.transform(X)
    pred = model.predict(X_processed)[0]
    prob = model.predict_proba(X_processed)[0].tolist()

    message = "⚠️ Khách hàng CÓ khả năng vỡ nợ!" if pred == 1 else "✅ Khách hàng ÍT khả năng vỡ nợ."

    return {
        "Kết quả": message,
        "Xác suất không vỡ nợ": f"{prob[0]*100:.2f}%",
        "Xác suất vỡ nợ": f"{prob[1]*100:.2f}%"
    }

# ==== Giao diện Gradio ====
inputs = [
    gr.Number(label="Hạn mức tín dụng (LIMIT_BAL)", value=200000),
    gr.Dropdown(choices=[1,2], label="Giới tính (1=Nam,2=Nữ)", value=2),
    gr.Dropdown(choices=[1,2,3,4], label="Trình độ học vấn", value=2),
    gr.Dropdown(choices=[1,2,3], label="Hôn nhân", value=1),
    gr.Number(label="Tuổi", value=35),

    gr.Number(label="PAY_1", value=0), gr.Number(label="PAY_2", value=0),
    gr.Number(label="PAY_3", value=0), gr.Number(label="PAY_4", value=0),
    gr.Number(label="PAY_5", value=0), gr.Number(label="PAY_6", value=0),

    gr.Number(label="BILL_AMT1", value=3913), gr.Number(label="BILL_AMT2", value=3102),
    gr.Number(label="BILL_AMT3", value=689), gr.Number(label="BILL_AMT4", value=0),
    gr.Number(label="BILL_AMT5", value=0), gr.Number(label="BILL_AMT6", value=0),

    gr.Number(label="PAY_AMT1", value=0), gr.Number(label="PAY_AMT2", value=0),
    gr.Number(label="PAY_AMT3", value=0), gr.Number(label="PAY_AMT4", value=0),
    gr.Number(label="PAY_AMT5", value=0), gr.Number(label="PAY_AMT6", value=0),
]

outputs = gr.JSON()

title = "💳 Dự đoán vỡ nợ khách hàng"
description = "Nhập thông tin khách hàng và nhấn **Dự đoán** để xem xác suất vỡ nợ."

demo = gr.Interface(
    fn=predict_default,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description
)

if __name__ == "__main__":
    demo.launch()
