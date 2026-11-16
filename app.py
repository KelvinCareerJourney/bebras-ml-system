import matplotlib
matplotlib.use('Agg')  # 關鍵：防止伺服器在無圖形介面下報錯
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestRegressor

# 模擬訓練一個示範用的 Random Forest 模型與特徵數據
np.random.seed(42)
X_dummy = np.random.rand(100, 4)
y_dummy = X_dummy @ np.array([10, 25, 15, 20]) + 50 + np.random.randn(100) * 2
feature_names = ['練習次數', '平均正確率', '總練習時長', '困難題答對率']

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_dummy, y_dummy)

def predict_and_compare(practice_count, accuracy, duration, hard_q_rate):
    # 1. 預測分數
    input_data = np.array([[practice_count/50, accuracy/100, duration/120, hard_q_rate/100]])
    score = float(model.predict(input_data)[0]) * 1.5 + 30
    score = min(max(score, 0), 100)

    # 2. 自動生成比較圖表：該名學生 vs 全校平均
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['練習次數', '平均正確率', '總練習時長', '困難題答對率']
    
    student_vals = [practice_count/50, accuracy/100, duration/120, hard_q_rate/100]
    school_avg = [0.5, 0.68, 0.45, 0.52] # 全校平均基準
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, student_vals, width, label='該名學生', color='#4f46e5')
    ax.bar(x + width/2, school_avg, width, label='全校平均', color='#94a3b8')
    
    ax.set_ylabel('表現標準化指數')
    ax.set_title('學生個人表現 vs 全校平均對比')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.legend()
    plt.tight_layout()
    
    return f"{score:.1f} 分", fig

def get_feature_importance_plot():
    # 3. 特徵重要性分析圖 (Feature Importance)
    fig, ax = plt.subplots(figsize=(6, 4))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    ax.barh(range(len(indices)), importances[indices], color='#10b981')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('相對決策權重')
    ax.set_title('Random Forest 模型特徵重要性分析')
    plt.tight_layout()
    return fig

# 建立多頁籤 (Tabs) 介面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏆 國際運算思維挑戰賽 (Bebras) - 學生表現預測與分析系統")
    
    with gr.Tabs():
        with gr.TabItem("📋 單筆預測與能力對比"):
            gr.Markdown("輸入學生的學習數據，即時計算預測評量分數，並產出與全校平均的對比長條圖。")
            with gr.Row():
                with gr.Column():
                    inp1 = gr.Number(label="練習次數 (次)", value=12)
                    inp2 = gr.Number(label="平均正確率 (%)", value=82)
                    inp3 = gr.Number(label="總練習時長 (分鐘)", value=45)
                    inp4 = gr.Number(label="困難題答對率 (%)", value=70)
                    btn = gr.Button("開始預測 🚀", variant="primary")
                with gr.Column():
                    out_score = gr.Textbox(label="🎯 預測評量分數")
                    out_plot = gr.Plot(label="學生 vs 全校平均對比圖")
            btn.click(predict_and_compare, inputs=[inp1, inp2, inp3, inp4], outputs=[out_score, out_plot])
            
        with gr.TabItem("📊 模型特徵重要性分析"):
            gr.Markdown("透過 Random Forest 機器學習模型檢視決策權重，證明「困難題答對率」與「平均正確率」對最終評分的關鍵影響力。")
            fi_btn = gr.Button("載入模型特徵重要性圖表", variant="secondary")
            fi_plot = gr.Plot(label="模型決策權重圖")
            fi_btn.click(get_feature_importance_plot, outputs=[fi_plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501)