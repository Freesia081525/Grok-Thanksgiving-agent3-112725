import streamlit as st
import json
from io import BytesIO
import pyyaml
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import pandas as pd
import markdown as md_lib
from datetime import datetime
import openai
import google.generativeai as genai
from anthropic import Anthropic
import httpx
import os
# ====================== 讀取外部 31 個繁體中文 Agent ======================
def load_agents(path: str = "agents.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = pyyaml.safe_load(f)
        return data.get("agents", [])
    except FileNotFoundError:
        st.warning("找不到 agents.yaml，請確認檔案存在於同一目錄。")
        return []
    except Exception as e:
        st.error(f"讀取 agents.yaml 失敗：{e}")
        return []

AGENTS = load_agents()

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
    "logs": [],
    "theme": "櫻花・粉櫻",
    "dark": True,
    "input_text": "",
    "ordered_agents": [],
    "chain_outputs": {},
    "docs": {},                 # doc_id -> {name, type, bytes, text, ocr_markdown, summary, entities}
    "active_doc_id": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ====================== 工具函式 ======================
def log(level: str, msg: str):
    st.session_state.logs.append({"t": datetime.now().strftime("%H:%M:%S"), "lvl": level, "msg": msg})

def call_llm(provider: str, model: str, key: str, messages, max_tokens=3000, temp=0.7):
    if not key or not key.strip():
        raise ValueError(f"{provider.upper()} API Key 不可為空")
    try:
        if provider == "openai":
            client = openai.OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp,
            )
            return resp.choices[0].message.content.strip()

        elif provider == "gemini":
            genai.configure(api_key=key)
            m = genai.GenerativeModel(model)
            # messages: [{"role":"user","content":"..."}]
            contents = []
            for msg in messages:
                contents.append({"role": msg["role"], "parts": [msg["content"]]})
            resp = m.generate_content(contents)
            return resp.text

        elif provider == "anthropic":
            client = Anthropic(api_key=key)
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temp,
                messages=messages,
            )
            return resp.content[0].text

        elif provider == "xai":
            resp = httpx.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                },
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"呼叫 {provider} 失敗：{str(e)}")

# ---- PDF / OCR 工具 ----
def parse_page_selection(selection: str, max_pages: int):
    """
    將使用者輸入如 '1-3,5' 轉為 [1,2,3,5]，頁碼從 1 開始。
    """
    pages = set()
    if not selection:
        return list(range(1, max_pages + 1))
    for part in selection.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            start, end = int(start), int(end)
            for p in range(start, end + 1):
                if 1 <= p <= max_pages:
                    pages.add(p)
        else:
            p = int(part)
            if 1 <= p <= max_pages:
                pages.add(p)
    return sorted(pages)

def pdf_to_images(pdf_bytes: bytes, page_numbers: list[int]):
    images = convert_from_bytes(pdf_bytes)
    # convert_from_bytes 產生從 1 開始的頁序，實際 list index 從 0
    result = []
    for p in page_numbers:
        if 1 <= p <= len(images):
            result.append((p, images[p - 1]))
    return result

def ocr_with_tesseract(images_with_index, lang="eng+chi_tra"):
    texts = []
    for page_num, img in images_with_index:
        text = pytesseract.image_to_string(img, lang=lang)
        texts.append(f"# Page {page_num}\n\n{text.strip()}\n")
    return "\n\n".join(texts)

def ocr_with_llm_openai(image: Image.Image, api_key: str, model: str, max_tokens: int):
    import base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "請以純文字（繁體中文/英文）完整轉錄這張圖片中的所有可見文字，不要加入額外解釋。",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return resp.choices[0].message.content.strip()

def summarize_and_extract_entities(openai_key: str, text: str, model: str = "gpt-4o-mini"):
    """
    回傳：summary_markdown (含 coral 關鍵字) 與 entities (list[dict])
    """
    # 1) 要求 LLM 產出 JSON（20 個實體與脈絡）
    entity_prompt = f"""
你是一位精準的資訊抽取專家，請從以下文字中抽取 **20 個最重要的實體**，用繁體中文輸出 JSON，結構如下：

{{
  "entities": [
    {{
      "id": "E1",
      "name": "實體名稱",
      "type": "類型（人物/組織/地點/概念/事件/工具/指標/其他）",
      "context": "簡要說明此實體在本文中的角色與重要性",
      "evidence": "引用原文或高度貼近原文的關鍵片段",
      "category": "主題分類（例如：技術、商業、教育、法律…）"
    }},
    ...
  ]
}}

請嚴格只輸出合法 JSON，不要加任何多餘文字。
待分析文字如下：
{text[:12000]}
"""
    entity_json_str = call_llm(
        "openai",
        model,
        openai_key,
        [{"role": "user", "content": entity_prompt}],
        max_tokens=4000,
        temp=0.2,
    )

    entities = []
    try:
        data = json.loads(entity_json_str)
        entities = data.get("entities", [])
    except Exception:
        # 解析失敗就當作空，並直接把原始字串放進一個實體
        entities = [
            {
                "id": "E1",
                "name": "解析失敗",
                "type": "系統",
                "context": "無法解析 LLM 回傳的 JSON，以下為原始內容。",
                "evidence": entity_json_str[:2000],
                "category": "錯誤",
            }
        ]

    # 2) 要求 LLM 產出含 coral 關鍵字的 Markdown 摘要
    summary_prompt = f"""
你是一位專業的文本總結與資訊設計師。請根據以下文字產出一份 **綜合性 Markdown 摘要**，要求：

1. 條列清楚、段落分明，使用標題 (##, ###) 與條列清單。
2. 自行判斷 10–20 個最關鍵的名詞或術語，並用 HTML 標記：
   `<span style="color:coral">關鍵詞</span>`。
3. 摘要需聚焦於：主題、關鍵觀點、重要人物/組織/事件、潛在風險與機會。

原始文字如下：
{text[:12000]}
"""
    summary_md = call_llm(
        "openai",
        model,
        openai_key,
        [{"role": "user", "content": summary_prompt}],
        max_tokens=2000,
        temp=0.4,
    )

    return summary_md, entities

# ====================== 主題 CSS + 深色模式 ======================
theme = THEMES[st.session_state.theme]
bg = theme["bgD"] if st.session_state.dark else theme["bgL"]
st.markdown(
    f"""
<style>
    :root {{ --primary: {theme['p']}; --secondary: {theme['s']}; }}
    .stApp {{ background: {bg}; color: {'#E8D9FF' if st.session_state.dark else '#333'}; }}
    .css-1d391kg, .css-1v0mbdj {{ background: transparent !important; }}
    .agent-item {{ padding: 8px; margin: 4px 0; background: rgba(255,255,255,0.1); border-radius: 8px; cursor: move; }}
    table td, table th {{ font-size: 0.9rem; }}
</style>
""",
    unsafe_allow_html=True,
)

# ====================== Sidebar ======================
with st.sidebar:
    st.markdown("### 花卉主題")
    st.session_state.theme = st.selectbox(
        "選擇主題",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
    )

    st.session_state.dark = st.checkbox("深色模式", value=st.session_state.dark)

    st.markdown("### API 金鑰")
    api_keys = {
        "openai": st.text_input(
            "OpenAI", type="password", value=os.getenv("OPENAI_API_KEY", "")
        ),
        "gemini": st.text_input(
            "Gemini", type="password", value=os.getenv("GEMINI_API_KEY", "")
        ),
        "xai": st.text_input(
            "xAI", type="password", value=os.getenv("XAI_API_KEY", "")
        ),
        "anthropic": st.text_input(
            "Anthropic", type="password", value=os.getenv("ANTHROPIC_API_KEY", "")
        ),
    }

    st.markdown("### 執行日誌")
    for entry in st.session_state.logs[-12:]:
        color = {
            "success": "lightgreen",
            "error": "salmon",
            "warning": "orange",
            "info": "lightblue",
        }.get(entry["lvl"], "gray")
        st.markdown(
            f"<small style='color:{color}'>[{entry['t']}] {entry['msg']}</small>",
            unsafe_allow_html=True,
        )

# ====================== 主標題 ======================
st.markdown(f"# {st.session_state.theme} AI 花園筆記本")
st.caption("文件上傳 + PDF 頁面 OCR + 31 個繁體中文專業分析 Agent • 多模型鏈式執行 • 20 個深度追問")

# ====================== Tabs ======================
tab_doc, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["文件 / OCR", "輸入文字", "多智能體鏈", "智能替換", "AI 筆記助手", "儀表板"]
)

# ====================== 0. 文件 / OCR 分頁 ======================
with tab_doc:
    st.subheader("文件上傳與預覽")

    uploaded_files = st.file_uploader(
        "上傳文件（txt, pdf, md, markdown, csv, json）",
        type=["txt", "pdf", "md", "markdown", "csv", "json"],
        accept_multiple_files=True,
    )

    # 讀入並存入 session_state.docs
    if uploaded_files:
        for f in uploaded_files:
            doc_id = f"{f.name}-{f.size}-{int(datetime.now().timestamp())}"
            if doc_id in st.session_state.docs:
                continue
            raw_bytes = f.read()
            ext = f.name.lower().split(".")[-1]

            doc_info = {
                "id": doc_id,
                "name": f.name,
                "type": ext,
                "bytes": raw_bytes,
                "text": "",
                "ocr_markdown": "",
                "summary": "",
                "entities": [],
            }

            # 預先解析純文字類型
            try:
                if ext in ["txt", "md", "markdown"]:
                    doc_info["text"] = raw_bytes.decode("utf-8", errors="ignore")
                    doc_info["ocr_markdown"] = doc_info["text"]

                elif ext == "csv":
                    df = pd.read_csv(BytesIO(raw_bytes))
                    doc_info["text"] = df.to_csv(index=False)
                    doc_info["ocr_markdown"] = "```csv\n" + doc_info["text"] + "\n```"

                elif ext == "json":
                    obj = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
                    pretty = json.dumps(obj, ensure_ascii=False, indent=2)
                    doc_info["text"] = pretty
                    doc_info["ocr_markdown"] = "```json\n" + pretty + "\n```"
            except Exception as e:
                log("warning", f"{f.name} 解析失敗：{e}")

            st.session_state.docs[doc_id] = doc_info
            st.session_state.active_doc_id = doc_id

        log("success", f"成功載入 {len(uploaded_files)} 個檔案")

    # 左側：文件清單；右側：預覽 + OCR + 分析
    col_list, col_main = st.columns([1, 3])

    with col_list:
        st.markdown("#### 文件清單")
        if not st.session_state.docs:
            st.info("尚未上傳任何文件")
        else:
            for doc_id, doc in st.session_state.docs.items():
                if st.button(doc["name"], key=f"doc_btn_{doc_id}"):
                    st.session_state.active_doc_id = doc_id

    with col_main:
        if not st.session_state.active_doc_id or st.session_state.active_doc_id not in st.session_state.docs:
            st.info("請從左側選擇一個文件")
        else:
            doc = st.session_state.docs[st.session_state.active_doc_id]
            st.markdown(f"### 當前文件：**{doc['name']}**")

            ext = doc["type"]

            with st.expander("文件預覽", expanded=True):
                if ext == "pdf":
                    # Streamlit 1.32+ 支援 st.pdf
                    st.pdf(BytesIO(doc["bytes"]))
                elif ext in ["txt", "md", "markdown"]:
                    st.text_area("原始文字", value=doc["text"], height=240)
                elif ext == "csv":
                    try:
                        df = pd.read_csv(BytesIO(doc["bytes"]))
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"CSV 預覽失敗：{e}")
                elif ext == "json":
                    try:
                        obj = json.loads(doc["bytes"].decode("utf-8", errors="ignore"))
                        st.json(obj)
                    except Exception as e:
                        st.error(f"JSON 預覽失敗：{e}")
                else:
                    st.write("暫不支援此格式的預覽。")

            # ===== PDF：頁碼選擇與 OCR 模式 =====
            if ext == "pdf":
                st.markdown("#### PDF OCR 設定")

                # 先算出總頁數（只轉一頁試探）
                try:
                    pages_tmp = convert_from_bytes(doc["bytes"], first_page=1, last_page=1)
                    # 以 first_page/last_page 選擇的方式重新估算總頁數：直接用 high last_page 會快些但實務再調
                    # 這裡簡化：先全部轉換一次再取 len
                    all_pages = convert_from_bytes(doc["bytes"])
                    total_pages = len(all_pages)
                except Exception as e:
                    st.error(f"PDF 解析失敗：{e}")
                    total_pages = 0
                    all_pages = []

                if total_pages > 0:
                    st.write(f"偵測到共 **{total_pages}** 頁")

                    page_sel = st.text_input(
                        "要 OCR 的頁碼（例如：1-3,5；空白代表全部）", value="1"
                    )
                    ocr_mode = st.radio(
                        "OCR 模式",
                        ["Python OCR（Tesseract：英/繁中）", "LLM OCR（多模態 GPT-4o）"],
                    )

                    if ocr_mode.startswith("Python"):
                        lang = st.selectbox(
                            "Tesseract 語言",
                            ["eng", "chi_tra", "eng+chi_tra"],
                            index=2,
                        )
                    else:
                        llm_model = st.text_input(
                            "OpenAI 多模態模型名稱", value="gpt-4o-mini"
                        )
                        llm_max_tokens = st.slider(
                            "每頁 Max Tokens", 256, 4096, 1024, step=128
                        )

                    if st.button("執行 OCR", type="primary"):
                        if not api_keys["openai"] and ocr_mode.startswith("LLM"):
                            st.error("請在側邊欄填入 OpenAI API Key")
                        else:
                            try:
                                with st.spinner("OCR 執行中，請稍候..."):
                                    sel_pages = (
                                        parse_page_selection(page_sel, total_pages)
                                        if page_sel.strip()
                                        else list(range(1, total_pages + 1))
                                    )

                                    # 取選定頁面的 image
                                    images_with_idx = [
                                        (i + 1, p)
                                        for i, p in enumerate(all_pages)
                                        if (i + 1) in sel_pages
                                    ]

                                    ocr_texts = []
                                    if ocr_mode.startswith("Python"):
                                        ocr_text = ocr_with_tesseract(
                                            images_with_idx, lang=lang
                                        )
                                    else:
                                        # LLM OCR for each page
                                        for page_num, img in images_with_idx:
                                            text = ocr_with_llm_openai(
                                                img,
                                                api_keys["openai"],
                                                llm_model,
                                                llm_max_tokens,
                                            )
                                            ocr_texts.append(
                                                f"# Page {page_num}\n\n{text}\n"
                                            )
                                        ocr_text = "\n\n".join(ocr_texts)

                                    doc["ocr_markdown"] = ocr_text
                                    doc["text"] = ocr_text
                                    st.session_state.docs[doc["id"]] = doc
                                    st.session_state.input_text = ocr_text
                                    log("success", f"{doc['name']} OCR 完成")
                                    st.success("OCR 完成，結果已寫入可編輯區與多智能體輸入。")
                            except Exception as e:
                                st.error(f"OCR 失敗：{e}")
                                log("error", f"OCR 失敗：{e}")

            # ===== OCR 結果（Markdown 可編輯） =====
            st.markdown("#### OCR / 文本結果（Markdown 可編輯）")
            ocr_md = st.text_area(
                "請在此修改 OCR 後的內容（支援 Markdown）",
                value=doc.get("ocr_markdown", ""),
                height=260,
                key=f"ocr_md_{doc['id']}",
            )
            doc["ocr_markdown"] = ocr_md
            doc["text"] = ocr_md
            st.session_state.docs[doc["id"]] = doc

            # 同步到主輸入文字，方便多智能體鏈使用
            if st.button("將此內容同步到『輸入文字』分頁"):
                st.session_state.input_text = ocr_md
                st.success("已同步到『輸入文字』分頁。")

            # ===== 摘要 + 實體抽取 =====
            st.markdown("#### 綜合摘要與 20 實體抽取")
            if st.button("產生摘要與 20 個實體（使用 OpenAI）"):
                if not api_keys["openai"]:
                    st.error("請在側邊欄填入 OpenAI API Key")
                elif not ocr_md.strip():
                    st.warning("目前沒有可分析的文字。")
                else:
                    try:
                        with st.spinner("AI 摘要與實體抽取中..."):
                            summary_md, entities = summarize_and_extract_entities(
                                api_keys["openai"], ocr_md
                            )
                            doc["summary"] = summary_md
                            doc["entities"] = entities
                            st.session_state.docs[doc["id"]] = doc
                            st.success("已完成摘要與實體抽取。")
                    except Exception as e:
                        st.error(f"產生摘要/實體失敗：{e}")
                        log("error", f"摘要/實體失敗：{e}")

            # 顯示摘要
            if doc.get("summary"):
                st.markdown("##### 綜合摘要（含珊瑚色關鍵字）")
                st.markdown(doc["summary"], unsafe_allow_html=True)

            # 顯示 20 實體：表格 + JSON
            if doc.get("entities"):
                st.markdown("##### 20 個實體與脈絡（表格）")
                df_entities = pd.DataFrame(doc["entities"])
                st.dataframe(df_entities, use_container_width=True)

                st.markdown("##### 20 個實體與脈絡（JSON）")
                st.json(doc["entities"])

            # ===== 從 agents.yaml 選擇 Agent 分析此文件 =====
            if AGENTS:
                st.markdown("#### 使用 Agent 分析此文件")
                agent_names = [a["name"] for a in AGENTS]
                selected_agents = st.multiselect(
                    "選擇要執行的 Agent",
                    options=agent_names,
                    default=agent_names[:5] if len(agent_names) >= 5 else agent_names,
                )

                if selected_agents and st.button("對此文件執行所選 Agent"):
                    if not api_keys["openai"] and not api_keys["gemini"] and not api_keys["anthropic"]:
                        st.error("請至少填入一個 API Key（OpenAI / Gemini / Anthropic）")
                    else:
                        for idx, agent_name in enumerate(selected_agents, start=1):
                            agent = next(a for a in AGENTS if a["name"] == agent_name)
                            model_cfg = agent.get("default_model", "OpenAI:gpt-4o-mini")
                            provider_key = model_cfg.split(":")[0]
                            model_name = model_cfg.split(":", 1)[1]

                            provider_map = {
                                "OpenAI": "openai",
                                "Gemini": "gemini",
                                "Anthropic": "anthropic",
                                "xAI": "xai",
                            }
                            provider = provider_map.get(provider_key, "openai")
                            key = api_keys.get(provider, "")

                            prompt_template = agent.get(
                                "prompt_template",
                                "你現在是「{name}」，請用繁體中文深入分析以下文本：\n\n{content}",
                            )
                            prompt = prompt_template.format(
                                name=agent["name"],
                                role=agent.get("role", ""),
                                description=agent.get("description", ""),
                                content=ocr_md,
                            )

                            st.markdown(f"##### {idx}. {agent_name}")
                            try:
                                with st.spinner(f"{agent_name} 分析中..."):
                                    out = call_llm(
                                        provider,
                                        model_name.strip(),
                                        key,
                                        [{"role": "user", "content": prompt}],
                                        max_tokens=agent.get("default_max_tokens", 2000),
                                        temp=agent.get("temperature", 0.7),
                                    )
                                    st.markdown(out)
                                    log("success", f"{agent_name} 完成（文件模式）")
                            except Exception as e:
                                st.error(str(e))
                                log("error", f"{agent_name} 失敗（文件模式）")

# ====================== 1. 輸入文字 ======================
with tab1:
    st.session_state.input_text = st.text_area(
        "請貼上要分析的文字", value=st.session_state.input_text, height=500
    )

# ====================== 2. 多智能體鏈（沿用原本邏輯，AGENTS 來自 agents.yaml） ======================
with tab2:
    if not st.session_state.input_text.strip():
        st.info("請先在「輸入文字」或「文件 / OCR」準備好內容")
    else:
        selected = st.multiselect(
            "選擇要執行的 Agent（可重複）",
            options=[a["name"] for a in AGENTS],
            default=[a["name"] for a in AGENTS[:6]],
        )

        if selected:
            if (
                "ordered_agents" not in st.session_state
                or st.session_state.ordered_agents != selected
            ):
                st.session_state.ordered_agents = selected.copy()

            st.markdown("### 調整執行順序")
            for i, agent_name in enumerate(st.session_state.ordered_agents):
                col1, col2, col3 = st.columns([6, 1, 1])
                with col1:
                    st.markdown(
                        f"<div class='agent-item'>#{i+1} {agent_name}</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    if i > 0 and st.button("↑", key=f"up{i}"):
                        s = st.session_state.ordered_agents
                        s[i], s[i - 1] = s[i - 1], s[i]
                        st.rerun()
                with col3:
                    if i < len(st.session_state.ordered_agents) - 1 and st.button(
                        "↓", key=f"down{i}"
                    ):
                        s = st.session_state.ordered_agents
                        s[i], s[i + 1] = s[i + 1], s[i]
                        st.rerun()

            col1a, col2a, col3a = st.columns(3)
            with col1a:
                default_model = st.selectbox(
                    "預設模型",
                    [
                        "OpenAI: gpt-4o-mini",
                        "Gemini: gemini-2.5-flash",
                        "Anthropic: claude-3-haiku-20240307",
                    ],
                    index=0,
                )
            with col2a:
                default_tokens = st.slider("預設 Max Tokens", 500, 8000, 3000)
            with col3a:
                default_temp = st.slider("預設 Temperature", 0.0, 1.0, 0.7, 0.05)

            current_text = st.session_state.input_text
            for idx, name in enumerate(st.session_state.ordered_agents):
                agent = next(a for a in AGENTS if a["name"] == name)
                with st.expander(f"{idx+1}. {name}", expanded=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        default_prompt = (
                            agent.get(
                                "prompt_template",
                                "你現在是「{name}」，請用繁體中文專業深入分析以下文字：\n\n{content}\n\n要求條理清晰、深度到位。",
                            )
                            .format(
                                name=agent["name"],
                                role=agent.get("role", ""),
                                description=agent.get("description", ""),
                                content=current_text,
                            )
                        )
                        prompt = st.text_area(
                            "提示詞（可編輯）",
                            height=160,
                            value=default_prompt,
                            key=f"prompt_{idx}",
                        )
                    with c2:
                        model_sel = st.selectbox(
                            "模型",
                            [
                                "OpenAI: gpt-4o-mini",
                                "Gemini: gemini-2.5-flash",
                                "Anthropic: claude-3-haiku-20240307",
                            ],
                            key=f"model_{idx}",
                        )
                        tokens = st.slider(
                            "Tokens",
                            500,
                            8000,
                            agent.get("default_max_tokens", default_tokens),
                            key=f"tok_{idx}",
                        )
                        temp = st.slider(
                            "Temp",
                            0.0,
                            1.0,
                            agent.get("temperature", default_temp),
                            0.05,
                            key=f"temp_{idx}",
                        )

                    if st.button(f"執行此步", key=f"run_{idx}"):
                        prov = {
                            "OpenAI": "openai",
                            "Gemini": "gemini",
                            "Anthropic": "anthropic",
                        }[model_sel.split(":")[0]]
                        mod = model_sel.split(": ")[-1]
                        try:
                            with st.spinner(f"{name} 分析中..."):
                                output = call_llm(
                                    prov,
                                    mod,
                                    api_keys[prov],
                                    [{"role": "user", "content": prompt}],
                                    max_tokens=tokens,
                                    temp=temp,
                                )
                                st.markdown(output)
                                current_text = st.text_area(
                                    "編輯後傳給下一 Agent",
                                    value=output,
                                    height=200,
                                    key=f"edit_{idx}",
                                )
                                log("success", f"{name} 完成")
                        except Exception as e:
                            st.error(str(e))
                            log("error", f"{name} 失敗")

            if st.button("生成 20 個深度追問問題", type="primary"):
                try:
                    q = call_llm(
                        "openai",
                        "gpt-4o-mini",
                        api_keys["openai"],
                        [
                            {
                                "role": "user",
                                "content": f"根據以上所有分析，生成 20 個極具哲學性與洞察力的追問問題（每題一行）：\n\n{current_text[:6000]}",
                            }
                        ],
                        max_tokens=1500,
                    )
                    st.markdown("### 20 個深度追問問題")
                    st.markdown(q.replace("\n", "  \n"))
                except Exception:
                    st.info("建議使用 OpenAI 產生追問")

# ====================== 3. 智能替換 ======================
with tab3:
    st.subheader("智能替換")
    word = st.text_input("想替換的詞")
    if word and st.button("產生 10 種更優美替代"):
        try:
            out = call_llm(
                "openai",
                "gpt-4o-mini",
                api_keys["openai"],
                [
                    {
                        "role": "user",
                        "content": f"給我 10 個比「{word}」更優美、文學感更強的繁體中文替代詞（每行一個）",
                    }
                ],
            )
            st.markdown(out.replace("\n", "  \n"))
        except Exception:
            st.error("請檢查 OpenAI Key")

# ====================== 4. AI 筆記助手 ======================
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        src = st.text_area("原始筆記", height=500)
    with col2:
        action = st.radio(
            "功能", ["AI 格式化", "關鍵字高亮", "實體表格", "心智圖 JSON"]
        )
        if st.button("執行"):
            try:
                result = call_llm(
                    "openai",
                    "gpt-4o-mini",
                    api_keys["openai"],
                    [
                        {
                            "role": "user",
                            "content": f"任務：{action}\n文字：{src}\n請用繁體中文回覆",
                        }
                    ],
                )
                if "JSON" in action:
                    st.json(result)
                else:
                    st.markdown(result, unsafe_allow_html=True)
            except Exception:
                st.error("請填 OpenAI Key")

# ====================== 5. 儀表板 ======================
with tab5:
    st.metric("文字長度", len(st.session_state.input_text))
    st.metric(
        "Agent 執行次數",
        len([l for l in st.session_state.logs if "完成" in l["msg"]]),
    )
    st.metric("已載入文件數量", len(st.session_state.docs))

log("success", "AI 花園筆記本已啟動（PyYAML + 文件 / OCR 版）")
