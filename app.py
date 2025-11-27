# app.py  —— 完全單檔、無第三方 UI 套件、原生 Streamlit 實作所有功能
import streamlit as st
import yaml
import json
import markdown
from datetime import datetime
import openai
import google.generativeai as genai
from anthropic import Anthropic
import httpx
import os

# ====================== 內建 31 個繁體中文 Agent ======================
AGENTS_YAML = """
agents:
  - name: "語意摘要大師"
    role: "將長文濃縮為 150 字內精準摘要，保留核心論點與情感"
  - name: "結構重組專家"
    role: "將雜亂文字重新組織為：引言、主論點、證據、結論 四段式結構"
  - name: "金句提取機"
    role: "找出文章中最有力量、最適合引用或分享的 5-10 句金句"
  - name: "情緒分析師"
    role: "分析全文情緒分布（正面/負面/中性），並標註高強度情緒段落"
  - name: "邏輯漏洞偵測"
    role: "找出論證中的邏輯謬誤、矛盾、前後不一致之處"
  - name: "立場傾向分析"
    role: "判斷作者立場（支持/反對/中立），並提供支持證據"
  - name: "關鍵人物關係圖"
    role: "提取人名、地名、組織，並建構關係網絡"
  - name: "事件時間軸整理"
    role: "將事件按時間順序整理成清晰年表"
  - name: "專業術語解釋"
    role: "列出所有專業術語並附上通俗解釋"
  - name: "比喻意象分析"
    role: "找出所有隱喻、比喻、象徵，並解釋其深層意涵"
  - name: "修辭手法盤點"
    role: "標記並解釋排比、對比、反問、設問等修辭"
  - name: "文化脈絡補充"
    role: "補充台灣/香港/大陸讀者可能需要的文化背景知識"
  - name: "潛台詞解讀"
    role: "讀出作者沒說出口，但暗示的意思"
  - name: "讀者反應預測"
    role: "預測不同族群讀者可能產生的共鳴或反彈"
  - name: "寫作風格診斷"
    role: "分析作者語氣（溫和/激進/幽默/冷峻）與用詞偏好"
  - name: "標題優化建議"
    role: "提供 5 個更吸睛、爭議性更高或更溫和的標題建議"
  - name: "社群爆紅潛力評分"
    role: "評分 1-10 分，預測這篇貼文在小紅書/Threads/Dcard 的爆紅機率"
  - name: "敏感詞檢測"
    role: "標記可能引發爭議或被檢舉的詞句（兩岸三地標準）"
  - name: "翻譯優化（簡轉繁）"
    role: "將簡體中文內容優美轉為繁體中文（台灣慣用語）"
  - name: "深度提問生成"
    role: "根據內容生成 10 個哲學級深度思考問題"
  - name: "反向觀點模擬"
    role: "模擬完全相反立場的人會如何反駁這篇文章"
  - name: "歷史類比分析"
    role: "將此事件與歷史上類似事件做對比"
  - name: "心理動機拆解"
    role: "分析作者寫這篇文章的潛在心理需求"
  - name: "段落精煉師"
    role: "將每一段濃縮為一句話，仍保留原意"
  - name: "觀點層次升級"
    role: "將普通觀點升級為更有洞見、更深刻的論述"
  - name: "跨領域連結"
    role: "連結心理學、社會學、哲學等理論來解讀內容"
  - name: "未來預測推論"
    role: "根據本文邏輯，推測未來可能發展的 3 種情境"
  - name: "讀後心得範本"
    role: "生成一篇 300 字讀後感（可直接發表）"
  - name: "迷因化改編"
    role: "將內容改編成 3 個適合製成迷因的版本"
  - name: "極簡總結（一圖勝千言）"
    role: "用一句話 + 一句話說明 + 一句話結論 總結全文"
  - name: "最終洞見提煉"
    role: "提煉出這篇文章真正想傳達的核心洞見（一句話）"
"""
AGENTS = yaml.safe_load(AGENTS_YAML)["agents"]

# ====================== 夢幻花卉主題（純 CSS）======================
THEMES = {
    "櫻花・粉櫻":   {"p": "#FF9CEE", "s": "#FFB7F3", "bgL": "#FFF5FB", "bgD": "#2D1B3A"},
    "薰衣草・紫夢": {"p": "#B19CD9", "s": "#C9A7EB", "bgL": "#F8F3FF", "bgD": "#2A1B3A"},
    "向日葵・金陽": {"p": "#FFB800", "s": "#FFD93D", "bgL": "#FFFBE6", "bgD": "#3A2F1B"},
    "玫瑰・紅艷":   {"p": "#E91E63", "s": "#FF6B9D", "bgL": "#FFF0F5", "bgD": "#3A1B2E"},
    "薄荷・清涼":   {"p": "#00D4AA", "s": "#4AEFCA", "bgL": "#F0FFF8", "bgD": "#1B3A38"},
    "夜空・星河":   {"p": "#6366F1", "s": "#818CF8", "bgL": "#EEF2FF", "bgD": "#0F172A"},
}

# ====================== Session State ======================
defaults = {
    "logs": [], "theme": "櫻花・粉櫻", "dark": True,
    "input_text": "", "ordered_agents": [], "chain_outputs": {}
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ====================== 工具函式 ======================
def log(level: str, msg: str):
    st.session_state.logs.append({"t": datetime.now().strftime("%H:%M:%S"), "lvl": level, "msg": msg})

def call_llm(provider: str, model: str, key: str, messages, max_tokens=3000, temp=0.7):
    if not key.strip():
        raise ValueError(f"{provider.upper()} API Key 不可為空")
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=key)
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temp)
            return resp.choices[0].message.content.strip()
        elif provider == "gemini":
            genai.configure(api_key=key)
            m = genai.GenerativeModel(model)
            resp = m.generate_content([m["content"] for m in messages])
            return resp.text
        elif provider == "anthropic":
            client = Anthropic(api_key=key)
            resp = client.messages.create(model=model, max_tokens=max_tokens, temperature=temp, messages=messages)
            return resp.content[0].text
        elif provider == "xai":
            resp = httpx.post("https://api.x.ai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp},
                              timeout=180)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"呼叫 {provider} 失敗：{str(e)}")

# ====================== 主題 CSS + 深色模式（純原生）======================
theme = THEMES[st.session_state.theme]
bg = theme["bgD"] if st.session_state.dark else theme["bgL"]
st.markdown(f"""
<style>
    :root {{ --primary: {theme['p']}; --secondary: {theme['s']}; }}
    .stApp {{ background: {bg}; color: {'#E8D9FF' if st.session_state.dark else '#333'}; }}
    .css-1d391kg, .css-1v0mbdj {{ background: transparent !important; }}
    .agent-item {{ padding: 8px; margin: 4px 0; background: rgba(255,255,255,0.1); border-radius: 8px; cursor: move; }}
</style>
""", unsafe_allow_html=True)

# ====================== Sidebar（原生實作所有 UI）======================
with st.sidebar:
    st.markdown("### 花卉主題")
    st.session_state.theme = st.selectbox("選擇主題", list(THEMES.keys()), 
                                         index=list(THEMES.keys()).index(st.session_state.theme))
    
    # 原生深色模式開關（用 checkbox + emoji）
    st.session_state.dark = st.checkbox("深色模式", value=st.session_state.dark)

    st.markdown("### API 金鑰")
    api_keys = {
        "openai": st.text_input("OpenAI", type="password", value=os.getenv("OPENAI_API_KEY","")),
        "gemini": st.text_input("Gemini", type="password", value=os.getenv("GEMINI_API_KEY","")),
        "xai": st.text_input("xAI", type="password", value=os.getenv("XAI_API_KEY","")),
        "anthropic": st.text_input("Anthropic", type="password", value=os.getenv("ANTHROPIC_API_KEY","")),
    }

    st.markdown("### 執行日誌")
    for entry in st.session_state.logs[-12:]:
        color = {"success":"lightgreen","error":"salmon","warning":"orange","info":"lightblue"}.get(entry["lvl"], "gray")
        st.markdown(f"<small style='color:{color}'>[{entry['t']}] {entry['msg']}</small>", unsafe_allow_html=True)

# ====================== 主標題 ======================
st.markdown(f"# {st.session_state.theme} AI 花園筆記本")
st.caption("31 個繁體中文專業分析 Agent • 多模型鏈式執行 • 20 個深度追問")

# ====================== Tabs ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["輸入文字", "多智能體鏈", "智能替換", "AI 筆記助手", "儀表板"])

# ====================== 1. 輸入文字 ======================
with tab1:
    st.session_state.input_text = st.text_area("請貼上要分析的文字", 
                                              value=st.session_state.input_text, height=500)

# ====================== 2. 多智能體鏈（純原生拖曳排序）=====================
with tab2:
    if not st.session_state.input_text.strip():
        st.info("請先在「輸入文字」貼上內容")
    else:
        # 選擇 Agent
        selected = st.multiselect("選擇要執行的 Agent（可重複）", 
                                  options=[a["name"] for a in AGENTS],
                                  default=[a["name"] for a in AGENTS[:6]])

        # 原生拖曳排序：使用 st.session_state + 上下箭頭按鈕
        if selected:
            if "ordered_agents" not in st.session_state or st.session_state.ordered_agents != selected:
                st.session_state.ordered_agents = selected.copy()

            st.markdown("### 拖曳調整執行順序")
            ordered = []
            for i, agent_name in enumerate(st.session_state.ordered_agents):
                col1, col2, col3 = st.columns([6, 1, 1])
                with col1:
                    st.markdown(f"<div class='agent-item'>#{i+1} {agent_name}</div>", unsafe_allow_html=True)
                with col2:
                    if i > 0 and st.button("↑", key=f"up{i}"):
                        st.session_state.ordered_agents[i], st.session_state.ordered_agents[i-1] = st.session_state.ordered_agents[i-1], st.session_state.ordered_agents[i]
                        st.rerun()
                with col3:
                    if i < len(st.session_state.ordered_agents)-1 and st.button("↓", key=f"down{i}"):
                        st.session_state.ordered_agents[i], st.session_state.ordered_agents[i+1] = st.session_state.ordered_agents[i+1], st.session_state.ordered_agents[i]
                        st.rerun()
                ordered.append(agent_name)

            # 預設參數
            col1, col2, col3 = st.columns(3)
            with col1:
                default_model = st.selectbox("預設模型", [
                    "OpenAI: gpt-4o-mini", "Gemini: gemini-2.5-flash", "Anthropic: claude-3-haiku-20240307"
                ], index=0)
            with col2:
                default_tokens = st.slider("預設 Max Tokens", 500, 8000, 3000)
            with col3:
                default_temp = st.slider("預設 Temperature", 0.0, 1.0, 0.7, 0.05)

            current_text = st.session_state.input_text
            for idx, name in enumerate(st.session_state.ordered_agents):
                agent = next(a for a in AGENTS if a["name"] == name)
                with st.expander(f"{idx+1}. {name}", expanded=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        prompt = st.text_area("提示詞（可編輯）", height=160,
                            value=f"你現在是「{name}」，請用繁體中文專業深入分析以下文字：\n\n{current_text}\n\n要求條理清晰、深度到位。",
                            key=f"prompt_{idx}")
                    with c2:
                        model_sel = st.selectbox("模型", [
                            "OpenAI: gpt-4o-mini", "Gemini: gemini-2.5-flash", "Anthropic: claude-3-haiku-20240307"
                        ], key=f"model_{idx}")
                        tokens = st.slider("Tokens", 500, 8000, default_tokens, key=f"tok_{idx}")
                        temp = st.slider("Temp", 0.0, 1.0, default_temp, 0.05, key=f"temp_{idx}")

                    if st.button(f"執行此步", key=f"run_{idx}"):
                        prov = {"OpenAI":"openai","Gemini":"gemini","Anthropic":"anthropic"}[model_sel.split(":")[0]]
                        mod = model_sel.split(": ")[-1]
                        try:
                            with st.spinner(f"{name} 分析中..."):
                                output = call_llm(prov, mod, api_keys[prov],
                                                 [{"role":"user","content":prompt}],
                                                 max_tokens=tokens, temp=temp)
                                st.markdown(output)
                                current_text = st.text_area("編輯後傳給下一 Agent", value=output, height=200, key=f"edit_{idx}")
                                log("success", f"{name} 完成")
                        except Exception as e:
                            st.error(str(e))
                            log("error", f"{name} 失敗")

            if st.button("生成 20 個深度追問問題", type="primary"):
                try:
                    q = call_llm("openai", "gpt-4o-mini", api_keys["openai"],
                                [{"role":"user","content":f"根據以上所有分析，生成 20 個極具哲學性與洞察力的追問問題（每題一行）：\n\n{current_text[:6000]}"}],
                                max_tokens=1500)
                    st.markdown("### 20 個深度追問問題")
                    st.markdown(q.replace("\n", "  \n"))
                except:
                    st.info("建議使用 OpenAI 產生追問")

# ====================== 其他分頁（簡化版）=====================
with tab3:  # 智能替換
    st.subheader("智能替換")
    word = st.text_input("想替換的詞")
    if word and st.button("產生 10 種更優美替代"):
        try:
            out = call_llm("openai","gpt-4o-mini",api_keys["openai"],
                          [{"role":"user","content":f"給我 10 個比「{word}」更優美、文學感更強的繁體中文替代詞（每行一個）"}])
            st.markdown(out.replace("\n", "  \n"))
        except: st.error("請檢查 OpenAI Key")

with tab4:  # AI 筆記助手（NoteKeeper）
    col1, col2 = st.columns(2)
    with col1:
        src = st.text_area("原始筆記", height=500)
    with col2:
        action = st.radio("功能", ["AI 格式化", "關鍵字高亮", "實體表格", "心智圖 JSON"])
        if st.button("執行"):
            try:
                result = call_llm("openai","gpt-4o-mini",api_keys["openai"],
                                 [{"role":"user","content":f"任務：{action}\n文字：{src}\n請用繁體中文回覆"}])
                if "JSON" in action:
                    st.json(result)
                else:
                    st.markdown(result, unsafe_allow_html=True)
            except: st.error("請填 OpenAI Key")

with tab5:  # 儀表板
    st.metric("文字長度", len(st.session_state.input_text))
    st.metric("Agent 執行次數", len([l for l in st.session_state.logs if "完成" in l["msg"]]))

log("success", "AI 花園筆記本已啟動（純原生版）")
