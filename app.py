import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import tempfile
import os

font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'sans-serif']

np.random.seed(42)
X_dummy = np.random.rand(300, 5)
y_dummy = 50 + (X_dummy[:, 0] * 35) + (X_dummy[:, 1] * 45) + (X_dummy[:, 2] * 70) + (X_dummy[:, 3] * 90) + (X_dummy[:, 4] * 20) + np.random.randn(300) * 2

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_dummy, y_dummy)

# 2025 年 Bebras 歷史 PR 參照：低 PR 區間依公告約略分數取中點，
# 並非精確的官方單點門檻；高分對照點依使用者確認的 210 分→PR90。
# 與 210 分 PR90 衝突的 215 分 PR89 暫不納入，待確認原公告。
PR_SCORES = np.array([112.5, 142.5, 172.5, 190, 200, 201, 210, 230, 245, 300], dtype=float)
PR_LEVELS = np.array([30, 50, 70, 81, 86, 86, 90, 93, 95, 99], dtype=float)

def estimated_pr(score):
    if not np.isfinite(score) or score < PR_SCORES[0]:
        return None
    return round(float(np.interp(score, PR_SCORES, PR_LEVELS)), 1)

def pr_display(score):
    value = estimated_pr(score)
    return f"約 PR {value:.1f}" if value is not None else "低於參照範圍"

def get_diagnostic_info(practice_count, duration, simple_acc, med_acc, hard_acc):
    badge = "🛡️ 穩紮穩打型"
    prescription = "同學，你的各項指標算滿平均的，表現中規中矩。建議可以開始挑戰多一點中等或困難題，突破舒適圈！"
    task = "🎯 前往 Google Classroom 完成「運算思維中級混合題庫」。"

    if practice_count >= 20 and simple_acc < 60:
        badge = "🧭 迷航探險家"
        prescription = "同學，我看到你練習了很多次（毅力值得嘉獎！），但簡單題答對率卻偏低，這代表你可能卡在基礎觀念，一直在用猜的。請先暫停盲目刷題！"
        task = "🎯 前往 Google Classroom 觀看「基礎運算思維觀念重塑」影片，釐清觀念後再重新挑戰簡單題。"
    elif practice_count <= 10:
        badge = "🌱 潛力新星"
        prescription = "同學，我發現你的練習次數偏少喔！運算思維是需要手感與經驗累積的，你的潛力還沒完全發揮，先從多做題目開始吧！"
        task = "🎯 前往練習網站，本週目標：完成 20 題基礎邏輯測驗。"
    elif hard_acc >= 75 and simple_acc <= 70:
        badge = "🌟 粗心小天才"
        prescription = "太可惜了！你的演算法拆解能力很強（困難題答對率高），但簡單題卻因為讀題太快而失分。建議作答時放慢速度，圈出題目關鍵字。"
        task = "🎯 前往 Google Classroom 開啟「閱讀理解與細節檢查特訓」專案。"
    elif practice_count >= 20 and simple_acc >= 75 and hard_acc <= 50:
        badge = "🦫 勤奮海狸"
        prescription = "同學，我發現你非常努力，基礎也不錯！但困難題一直遇到瓶頸。這代表你可能在「演算法拆解」的觀念上卡關了，先將核心邏輯重新梳理一次！"
        task = "🎯 前往 Google Classroom 8 年級資訊科技專區，開啟「Scratch 迴圈與陣列邏輯特訓」專案。"
    elif simple_acc >= 85 and med_acc >= 80 and hard_acc >= 75:
        badge = "👑 運算思維大師"
        prescription = "太厲害了！你的各項指標都非常優異，基本上已經具備 Bebras 挑戰賽頂標的水準。接下來請維持手感，並嘗試挑戰歷屆高難度魔王題！"
        task = "🎯 挑戰進階任務：前往高中部專區，試試看 Senior 組的挑戰題！"

    return badge, prescription, task

def generate_plot(simple_acc, med_acc, hard_acc):
    plt.figure(figsize=(8, 4))
    categories = ['簡單題', '中等題', '困難題']
    student_scores = [simple_acc, med_acc, hard_acc]
    school_avg = [85, 65, 40]  # 尚未串接實際全校平均；固定示意參考值
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, student_scores, width, label='該名學生', color='#4f46e5')
    plt.bar(x + width/2, school_avg, width, label='示意參考值', color='#cbd5e1')
    
    plt.ylabel('答對率 (%)')
    plt.title('各難易度題型答對率 vs 示意參考值對比')
    plt.xticks(x, categories)
    plt.ylim(0, 100)
    plt.legend()
    
    for i, v in enumerate(student_scores):
        plt.text(i - width/2, v + 2, f"{v}%", ha='center', color='#4f46e5', fontweight='bold')
    for i, v in enumerate(school_avg):
        plt.text(i + width/2, v + 2, f"{v}%", ha='center', color='gray')
        
    plt.tight_layout()
    plot_path = tempfile.mktemp(suffix=".png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def predict_single(practice_count, duration, simple_acc, med_acc, hard_acc):
    count_c = np.clip(practice_count, 0, 50)
    dur_c = np.clip(duration, 0, 120)
    simple_c = np.clip(simple_acc, 0, 100)
    med_c = np.clip(med_acc, 0, 100)
    hard_c = np.clip(hard_acc, 0, 100)

    features = np.array([[count_c/50.0, simple_c/100.0, med_c/100.0, hard_c/100.0, dur_c/120.0]])
    score = np.clip(model.predict(features)[0], 60, 300)
    pr = estimated_pr(score)
    
    badge, prescription, task = get_diagnostic_info(count_c, dur_c, simple_c, med_c, hard_c)
    plot_path = generate_plot(simple_c, med_c, hard_c)
    
    score_html = f"""
    <div style="background: linear-gradient(135deg, #4f46e5, #2563eb); color: white; padding: 20px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 14px; opacity: 0.9;">🎯 預測評量總分</div>
            <div style="font-size: 36px; font-weight: bold;">{score:.1f} <span style="font-size: 18px; font-weight: normal;">/300</span></div>
        </div>
        <div style="border-left: 2px solid rgba(255,255,255,0.2); padding-left: 20px;">
            <div style="font-size: 14px; opacity: 0.9;">📊 2025 歷史 PR 參照估計</div>
            <div style="font-size: 36px; font-weight: bold;">{pr_display(score)}</div>
        </div>
    </div>
    """
    return score_html, badge, prescription, task, plot_path

def batch_predict(file):
    try:
        df = pd.read_csv(file.name)
        required_cols = ['練習次數', '總練習時長', '簡單題答對率', '中等題答對率', '困難題答對率']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要欄位: {col}")
        
        features = np.zeros((len(df), 5))
        features[:, 0] = np.clip(df['練習次數'], 0, 50) / 50.0
        features[:, 1] = np.clip(df['簡單題答對率'], 0, 100) / 100.0
        features[:, 2] = np.clip(df['中等題答對率'], 0, 100) / 100.0
        features[:, 3] = np.clip(df['困難題答對率'], 0, 100) / 100.0
        features[:, 4] = np.clip(df['總練習時長'], 0, 120) / 120.0
        
        scores = np.clip(model.predict(features), 60, 300)
        df['預測總分'] = np.round(scores, 1)
        df['估計_PR'] = [estimated_pr(s) for s in scores]
        df['PR估計說明'] = ['低於歷史參照範圍' if estimated_pr(s) is None else '依2025年歷史對照點估計' for s in scores]
        
        badges, tasks = [], []
        for i in range(len(df)):
            b, _, t = get_diagnostic_info(
                df['練習次數'].iloc[i], df['總練習時長'].iloc[i],
                df['簡單題答對率'].iloc[i], df['中等題答對率'].iloc[i], df['困難題答對率'].iloc[i]
            )
            badges.append(b)
            tasks.append(t)
            
        df['獲得稱號'] = badges
        df['AI建議指派任務'] = tasks
        
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(output_file.name, index=False, encoding='utf-8-sig')
        return df, output_file.name
    except Exception as e:
        raise gr.Error(str(e))

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("""
    # 🏆 國際運算思維挑戰賽 (Bebras) - 學生表現預測與分析系統
    探索性估計學生學習表現、支援全班批量資料匯入，並結合遊戲化標籤與自適應派題，提供專屬 AI 診斷處方籤。
    """)
    
    with gr.Tabs():
        with gr.Tab("📝 單筆預測與 AI 診斷"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("### 📝 輸入學生學習數據")
                    
                    with gr.Row():
                        simple_acc = gr.Number(label="簡單題答對率 (%)", value=95, precision=0)
                        med_acc = gr.Number(label="中等題答對率 (%)", value=78, precision=0)
                        hard_acc = gr.Number(label="困難題答對率 (%)", value=45, precision=0)
                        
                    with gr.Row():
                        practice_count = gr.Number(label="練習次數 (次)", value=32, precision=0)
                        duration = gr.Number(label="總練習時長 (分鐘)", value=50, precision=0)
                        
                    predict_btn = gr.Button("開始預測與診斷 🚀", variant="primary")
                    score_out = gr.HTML()
                    
                with gr.Column(scale=6):
                    gr.Markdown("### 🤖 AI 專屬教練診斷報告")
                    with gr.Row():
                        badge_out = gr.Textbox(label="遊戲化稱號", interactive=False)
                        task_out = gr.Textbox(label="推薦任務", interactive=False)
                    prescription_out = gr.Textbox(label="🩺 學習處方籤", interactive=False, lines=3)
                    plot_out = gr.Image(label="學習落差對比分析")
                    
            predict_btn.click(
                fn=predict_single,
                inputs=[practice_count, duration, simple_acc, med_acc, hard_acc],
                outputs=[score_out, badge_out, prescription_out, task_out, plot_out]
            )
            
        with gr.Tab("📁 檔案批量上傳與自動派題"):
            with gr.Row():
                with gr.Column(scale=4):
                    file_input = gr.File(label="上傳全班 CSV 紀錄檔", file_types=['.csv'])
                    batch_btn = gr.Button("開始批量運算與診斷", variant="primary")
                    file_output = gr.File(label="下載完整診斷報表")
                with gr.Column(scale=8):
                    df_output = gr.Dataframe(label="預測結果預覽 (含 AI 自動派題)")
            
            batch_btn.click(fn=batch_predict, inputs=file_input, outputs=[df_output, file_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501)