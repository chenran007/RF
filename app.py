import streamlit as st

import joblib

import pandas as pd

import numpy as np

import shap

import matplotlib.pyplot as plt

model = joblib.load('RF.pkl')

X_test = pd.read_csv('X_test.csv')

#定义特征名称，对应数据集中的列名
feature_names = ["BC","YiDC", "PDC", "Age", "Pension", "WHtR", "CO", "BMI", "Smoking", "SCL", "Sleepquality", "Pain", "Eyesight", "Diffaction", "Hyperlipidemia", "Hyperuricemia","FLD", "OA", "Diabetes"]

#Streamlit 用户界面
st.title("老年高血压预测器")

BC = st.selectbox("平和质类型 (BC):", options=[0, 1])

YiDC = st.selectbox("阴虚质类型 (YiDC):", options=[0, 1])

PDC = st.selectbox("痰湿质类型 (PDC):", options=[0, 1])

Age = st.selectbox("年龄 (Age):", options=[0, 1,2])

Pension = st.selectbox("医保类型 (Pension):", options=[0, 1])

WHtR = st.selectbox("腰高比 (WHtR):", options=[0, 1])

CO = st.selectbox("中心性肥胖 (CO):", options=[0, 1])

BMI = st.selectbox("体质指数 (BMI):", options=[0,1,2,3])

Smoking = st.selectbox("吸烟 (Smoking):", options=[0, 1])

SCL = st.selectbox("精神文化生活 (SCL):", options=[0, 1])

Sleepquality = st.selectbox("睡眠质量 (Sleepquality):", options=[0,1,2,3])

Pain = st.selectbox("身体疼痛 (Pain):", options=[0, 1])

Eyesight = st.selectbox("视力 (Eyesight):", options=[0, 1,2,3])

Diffaction = st.selectbox("行动困难 (Diffaction):", options=[0, 1])

Hyperlipidemia = st.selectbox("高脂血症 (Hyperlipidemia):", options=[0, 1])

Hyperuricemia = st.selectbox("高尿酸血症 (Hyperuricemia):", options=[0, 1])

FLD = st.selectbox("脂肪肝 (FLD):", options=[0, 1])

OA = st.selectbox("关节炎 (OA):", options=[0, 1])

Diabetes = st.selectbox("糖尿病 (Diabetes):", options=[0, 1])

# 实现输入数据并进行预测
feature_values = [BC,YiDC, PDC, Age, Pension, WHtR, CO, BMI, Smoking, SCL, Sleepquality, Pain, Eyesight, Diffaction, Hyperlipidemia, Hyperuricemia,FLD, OA, Diabetes]  # 将用户输入的特征值存入列表
features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 预测类别（0: 无高血压，1: 有高血压）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    # 如果预测类别为 1（高风险）
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of hypertension. "
            f"The model predicts that your probability of having hypertension is {probability:.1f}%. "
            "It's advised to consult with your healthcare provider for further evaluation and possible intervention."
        )
   # 如果预测类别为 0（低风险）
    else:
        advice = (
            f"According to our model, you have a low risk of hypertension. "
            f"The model predicts that your probability of not having hypertension is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."
        )
    st.write(advice)
    # SHAP 解释
    st.subheader("SHAP Force Plot Explanation")
    # 创建 SHAP 解释器，基于树模型（如随机森林）
    explainer_shap = shap.TreeExplainer(model)
    # 计算 SHAP 值，用于解释模型的预测
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 根据预测类别显示 SHAP 强制图
    # 期望值（基线值）
    # 解释类别 1（患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:, :, 1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    # 期望值（基线值）
    # 解释类别 0（未患病）的 SHAP 值
    # 特征值数据
    # 使用 Matplotlib 绘图
    else:
         shap.force_plot(explainer_shap.expected_value[0], shap_values[:, :, 0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
