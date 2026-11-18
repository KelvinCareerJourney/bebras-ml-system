import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import tempfile
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestRegressor

font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
chinese_font = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
X_dummy = np.random.rand(300, 5)
y_dummy = 50 + (X_dummy[:, 0] * 35) + (X_dummy[:, 1] * 45) + (X_dummy[:, 2] * 70) + (X_dummy[:, 3] * 90) + (X_dummy[:, 4] * 20) + np.random.randn(300) * 2
feature_names = ["練習次數", "簡單題答對率", "中等題答對率", "困難題答對率", "總練習時長"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_dummy, y_dummy)

# PR 參照：來源為使用者提供的 2025 年 PR 公告／分數參照截圖。
# 高分區對照（按截圖所列）：190→81，200/201→86，210→90，230→93，245→95，300→99。
# 低分區只有「約」的分數區間，使用區間中點作為暫定近似錨點，不能視為精確官方單點常模。
# 以使用者確認的「210 分→PR90」為準；截圖中的 215 分→PR89 不列入換算。
PR_SCORES = np.array([112.5, 142.5, 172.5, 190, 200, 201, 210, 230, 245, 300], dtype=float)
PR_LEVELS = np.array([30, 50, 70, 81, 86, 86, 90, 93, 95, 99], dtype=float)

def estimated_pr(score):
    """歷史分數／PR 參照估計；低於最小錨點不作虛構外推。"""
    score = float(score)
    if not np.isfinite(score) or score < PR_SCORES[0]:
        return None
    return float(np.interp(score, PR_SCORES, PR_LEVELS))

def pr_label(score):
    value = estimated_pr(score)
    return f"約 PR {value:.1f}" if value is not None else "低於參照範圍"

def get_diagnostic_info(practice_count, easy, mid, hard, duration, score):
    badges, badge_text = [], []
    if hard >= 70 and easy < 70:
        badges.append("<span style='background:#fef3c7; color:#d97706; padding:5px 12px; border-radius:20px; font-size:12px; font-weight:700;'>🌟 粗心小天才</span>")
        badge_text.append("🌟 粗心小天才")
    if easy >= 80 and mid >= 70 and hard < 50:
        badges.append("<span style='background:#d1fae5; color:#059669; padding:5px 12px; border-radius:20px; font-size:12px; font-weight:700;'>🛡️ 穩紮穩打型</span>")
        badge_text.append("🛡️ 穩紮穩打型")
    if duration <= 45 and score >= 200:
        badges.append("<span style='background:#ede9fe; color:#7c3aed; padding:5px 12px; border-radius:20px; font-size:12px; font-weight:700;'>🚀 演算法潛力股</span>")
        badge_text.append("🚀 演算法潛力股")
    if score >= 240:
        badges.append("<span style='background:#fee2e2; color:#dc2626; padding:5px 12px; border-radius:20px; font-size:12px; font-weight:700;'>👑 運算思維大師</span>")
        badge_text.append("👑 運算思維大師")
    if practice_count >= 30:
        badges.append("<span style='background:#dbeafe; color:#2563eb; padding:5px 12px; border-radius:20px; font-size:12px; font-weight:700;'>🦫 勤奮海狸</span>")
        badge_text.append("🦫 勤奮海狸")
    if not badges:
        badges.append("<span style='background:#f1f5f9; color:#475569; padding:5px 12px; border-radius:20px; font-size:12px; font-weight:700;'>🌱 運算新手村</span>")
        badge_text.append("🌱 運算新手村")

    if practice_count >= 30 and hard < 50:
        prescription = "同學，我發現你非常努力（練習次數達標），基礎也不錯！但困難題一直遇到瓶頸。這代表你可能在「演算法拆解」的觀念上卡關了，建議暫時停止盲目刷題，先將核心邏輯重新梳理一次！"
        action = "請前往 Google Classroom 8 年級資訊科技專區，開啟「Scratch 迴圈與陣列邏輯特訓」專案，把核心積木重新拼一次。"
    elif hard >= 70 and easy < 70:
        prescription = "你的邏輯推理能力非常強，連困難題都能迎刃而解！但簡單題的答對率反而偏低，這通常是因為「粗心」或「讀題太快」造成的。"
        action = "請回到 Google Classroom 完成 10 題基礎觀念限時回測，強迫自己放慢作答速度，挑戰 100% 正確率。"
    elif easy < 60:
        prescription = "目前在基礎運算思維的觀念上還有些不熟悉，不要灰心！萬丈高樓平地起，先把基礎打穩最重要。"
        action = "請開啟 Scratch 3.0，重新實作一次「變數與基本運算」的課堂範例，確認每塊積木的意義。"
    elif score >= 245:
        prescription = "太棒了！你目前預測分數達到所提供歷史 PR95 對照分數，各項難易度的掌握度都非常優異。"
        action = "保持手感！建議開始挑戰我們課堂上的「進階星號題」，並嘗試將解題過程寫成 Scratch 程式碼來驗證思維。"
    else:
        prescription = "你的表現相當穩定！中等難度的邏輯轉換是目前拉高分數的關鍵，可以試著把答對率再往上推進。"
        action = "本週重點請放在「中等題型」的針對性練習，請登入平台完成本週派發的「Bebras 邏輯推理進階任務」。"

    return "".join(badges), "、".join(badge_text), prescription, action

def predict_and_compare(practice_count, easy_acc, mid_acc, hard_acc, duration):
    input_data = np.array([[min(practice_count, 50)/50, easy_acc/100, mid_acc/100, hard_acc/100, min(duration, 120)/120]])
    score = min(max(float(model.predict(input_data)[0]), 60), 300)
    pr_text = pr_label(score)

    fig, ax = plt.subplots(figsize=(9.5, 3.3), dpi=110)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    categories = ["簡單題", "中等題", "困難題"]
    student_vals = [easy_acc/100, mid_acc/100, hard_acc/100]
    avg_vals = [0.85, 0.65, 0.40]
    x = np.arange(len(categories))
    width = 0.28

    rects1 = ax.bar(x - width/2, student_vals, width, label="該名學生", color="#4f46e5", zorder=3)
    rects2 = ax.bar(x + width/2, avg_vals, width, label="全校平均", color="#cbd5e1", zorder=3)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{int(h*100)}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#4f46e5")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{int(h*100)}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#64748b")

    ax.set_title("各難易度題型答對率 vs 全校平均對比", fontproperties=chinese_font, fontsize=12, fontweight="bold", color="#1e293b", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontproperties=chinese_font, fontsize=11, color="#334155")
    ax.set_ylim(0, 1.18)
    ax.legend(prop=chinese_font, frameon=True, facecolor="#f8fafc", edgecolor="#e2e8f0", fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.14)

    html_score = f"""
    <div style="background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%); padding: 20px 18px; border-radius: 14px; color: white; margin-top: 12px; box-shadow: 0 10px 20px -5px rgba(79, 70, 229, 0.28); display: flex; justify-content: space-around; align-items: center;">
        <div style="text-align: center;">
            <div style="font-size: 11px; opacity: 0.85; font-weight: 600; letter-spacing: 0.5px;">🎯 預測評量總分</div>
            <div style="font-size: 32px; font-weight: 800; margin-top: 4px; letter-spacing: -1px;">{score:.1f} <span style="font-size: 14px; font-weight: 500; opacity: 0.85;">/ 300</span></div>
        </div>
        <div style="width: 1px; height: 48px; background: rgba(255,255,255,0.25);"></div>
        <div style="text-align: center;">
            <div style="font-size: 11px; opacity: 0.85; font-weight: 600; letter-spacing: 0.5px;">📊 歷史 PR 參照估計</div>
            <div style="font-size: 28px; font-weight: 800; margin-top: 4px;">{pr_text}</div>
        </div>
    </div>
    """
    b_html, _, pres, act = get_diagnostic_info(practice_count, easy_acc, mid_acc, hard_acc, duration, score)
    html_diagnostic = f"""
    <div style="background: white; padding: 20px 22px; border-radius: 14px; border: 1px solid #f1f5f9; box-shadow: 0 4px 16px -2px rgba(0,0,0,0.03);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 14px;">
            <div style="font-size: 15px; font-weight: 800; color: #1e293b;">🤖 AI 專屬教練診斷報告</div>
            <div style="display: flex; flex-wrap: wrap; gap: 6px;">{b_html}</div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div style="background: #f8fafc; padding: 12px 14px; border-radius: 10px; border-left: 3px solid #4f46e5;">
                <div style="font-size: 12px; font-weight: 700; color: #4f46e5; margin-bottom: 4px;">📝 學習處方籤</div>
                <div style="color: #334155; font-size: 13px; line-height: 1.45;">{pres}</div>
            </div>
            <div style="background: #fffbeb; padding: 12px 14px; border-radius: 10px; border-left: 3px solid #f59e0b;">
                <div style="font-size: 12px; font-weight: 700; color: #d97706; margin-bottom: 4px;">🎯 推薦任務</div>
                <div style="color: #92400e; font-size: 13px; line-height: 1.45;">{act}</div>
            </div>
        </div>
    </div>
    """
    return html_score, html_diagnostic, fig

def batch_predict(file):
    if file is None:
        return None, None
    try:
        df = pd.read_csv(file.name)
        required_cols = ["練習次數", "簡單題答對率", "中等題答對率", "困難題答對率", "總練習時長"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要欄位：{col}")
        norm_features = df[required_cols].values / np.array([50.0, 100.0, 100.0, 100.0, 120.0])
        scores = np.clip(model.predict(norm_features), 60, 300)
        prs = [estimated_pr(s) for s in scores]
        df["預測總分"] = np.round(scores, 1)
        df["歷史PR估計值"] = pd.Series(prs, index=df.index, dtype=float).round(1)
        df["PR估計說明"] = [
            "低於參照範圍" if p is None else "依歷史對照點估計；低 PR 區間為近似值"
            for p in prs
        ]
        badges_list, action_list = [], []
        for i, row in df.iterrows():
            _, b_plain, _, act = get_diagnostic_info(row["練習次數"], row["簡單題答對率"], row["中等題答對率"], row["困難題答對率"], row["總練習時長"], df.at[i, "預測總分"])
            badges_list.append(b_plain)
            action_list.append(act)
        df["獲得稱號 (系統判定)"] = badges_list
        df["AI 建議指派任務"] = action_list
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        return df, output_file
    except Exception as e:
        raise gr.Error(f"處理檔案時發生錯誤：{str(e)}")

def get_feature_importance_plot():
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=100)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    importances = model.feature_importances_
    indices = np.argsort(importances)
    ax.barh(range(len(indices)), importances[indices], color="#10b981", alpha=0.95, height=0.45, zorder=3)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontproperties=chinese_font, fontsize=10, color="#475569")
    ax.set_xlabel("相對決策權重", fontproperties=chinese_font, fontsize=9, color="#64748b", fontweight="500")
    ax.set_title("Random Forest 模型特徵重要性分析（含練習次數增益）", fontproperties=chinese_font, fontsize=11, fontweight="bold", color="#1e293b", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#f1f5f9")
    ax.spines["bottom"].set_color("#f1f5f9")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    plt.subplots_adjust(left=0.14, right=0.96, top=0.85, bottom=0.15)
    return fig

custom_css = """
body { background-color: #f8fafc !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; }
.gradio-container { max-width: 1380px !important; margin: auto; padding-top: 18px; padding-bottom: 36px; }
.main-header { background: white; padding: 18px 26px; border-radius: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 14px; border: 1px solid #f1f5f9; }
.card-box { background: white !important; padding: 20px 22px !important; border-radius: 14px !important; box-shadow: 0 4px 16px -2px rgba(0,0,0,0.03) !important; border: 1px solid #f1f5f9 !important; }
.gr-input, .gr-number input, textarea { background-color: #f8fafc !important; border: 1px solid #e2e8f0 !important; border-radius: 8px !important; padding: 6px 10px !important; }
button.primary { background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important; }
.gradio-plot { padding: 0 !important; border: none !important; box-shadow: none !important; background: transparent !important; width: 100% !important; }
.gradio-plot img, .gradio-plot svg { width: 100% !important; max-width: 100% !important; height: auto !important; display: block !important; }
"""

with gr.Blocks(title="Bebras 學生表現預測與分析系統", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"), css=custom_css) as demo:
    with gr.Column(elem_classes="main-header"):
        gr.Markdown("# 🏆 國際運算思維挑戰賽 (Bebras) - 學生表現預測與分析系統\n探索性估計學生學習表現、支援全班批量資料匯入，並結合**遊戲化標籤**與**自適應派題**，提供專屬 AI 診斷處方籤。")
    with gr.Tabs():
        with gr.TabItem("📋 單筆預測與 AI 診斷"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, elem_classes="card-box"):
                    gr.Markdown("### 📝 輸入學生學習數據")
                    with gr.Row():
                        inp1 = gr.Number(label="練習次數 (次)", value=32)
                        inp5 = gr.Number(label="總練習時長 (分鐘)", value=50)
                        inp2 = gr.Number(label="簡單題答對率 (%)", value=95)
                    with gr.Row():
                        inp3 = gr.Number(label="中等題答對率 (%)", value=78)
                        inp4 = gr.Number(label="困難題答對率 (%)", value=45)
                    btn = gr.Button("開始預測與診斷 🚀", variant="primary", size="lg")
                    out_score = gr.HTML()
                with gr.Column(scale=6):
                    out_diag = gr.HTML()
                    with gr.Column(elem_classes="card-box"):
                        out_plot = gr.Plot(show_label=False)
            demo.load(predict_and_compare, inputs=[inp1, inp2, inp3, inp4, inp5], outputs=[out_score, out_diag, out_plot])
            btn.click(predict_and_compare, inputs=[inp1, inp2, inp3, inp4, inp5], outputs=[out_score, out_diag, out_plot])
        with gr.TabItem("📂 檔案批量上傳與自動派題"):
            with gr.Column(elem_classes="card-box"):
                gr.Markdown("### 📥 上傳學生資料 CSV 檔案\n系統將估計全班總分與歷史 PR 參照值，並直接在報表中為每位學生**貼上遊戲化稱號**與**指派 Google Classroom / Scratch 專屬補救任務**！")
                file_input = gr.File(label="選擇 CSV 檔案", file_types=[".csv"])
                batch_btn = gr.Button("開始批量運算 ⚡", variant="primary")
                gr.Markdown("### 📊 批量計算結果總表")
                out_table = gr.Dataframe(label="全班預測與診斷總表")
                download_btn = gr.File(label="下載完整預測結果 CSV")
                batch_btn.click(batch_predict, inputs=[file_input], outputs=[out_table, download_btn])
        with gr.TabItem("📊 模型特徵重要性分析"):
            with gr.Column(elem_classes="card-box"):
                gr.Markdown("透過 Random Forest 機器學習模型檢視各項學習特徵的決策權重，驗證練習次數與各難易度答對率對最終評分的影響力。")
                fi_btn = gr.Button("載入模型特徵重要性圖表", variant="secondary")
                fi_plot = gr.Plot(show_label=False)
                fi_btn.click(get_feature_importance_plot, outputs=[fi_plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501)
