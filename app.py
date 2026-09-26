import streamlit as st
import pandas as pd
import urllib.parse
from transformers import pipeline
from pythainlp.tag import NER

# --- 1. ตั้งค่าหน้าตา Web App ---
st.set_page_config(
    page_title="AI Fact-Checker & Reference Verification",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI News Fact-Checker & Trust Verification System")
st.caption("ระบบวิเคราะห์ความน่าเชื่อถือของข่าว พร้อมโปร่งใสด้วยเหตุผลและแหล่งอ้างอิงตรวจสอบ (Explainable AI)")

# --- 2. โหลดโมเดล AI ---
@st.cache_resource
def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis", 
        model="poom-sci/WangchanBERTa-finetuned-sentiment"
    )
    classifier_model = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli"
    )
    return sentiment_model, classifier_model

with st.spinner("กำลังเตรียมระบบวิเคราะห์และค้นหาแหล่งอ้างอิง..."):
    sentiment_pipe, classifier_pipe = load_models()
    ner_engine = NER()

# --- ฟังก์ชันจัดการ NER ภาษาไทย ---
def combine_ner_tokens(ner_tags):
    combined_entities = []
    current_word = ""
    current_tag = ""

    for word, tag in ner_tags:
        if tag == 'O':
            if current_word:
                combined_entities.append((current_word, current_tag))
                current_word = ""
                current_tag = ""
            continue
        
        tag_type = tag.split('-')[-1]
        
        if tag.startswith('B-'):
            if current_word:
                combined_entities.append((current_word, current_tag))
            current_word = word
            current_tag = tag_type
        elif tag.startswith('I-') and tag_type == current_tag:
            current_word += word
        else:
            if current_word:
                combined_entities.append((current_word, current_tag))
            current_word = word
            current_tag = tag_type
            
    if current_word:
        combined_entities.append((current_word, current_tag))
        
    clean_entities = []
    prefixes = ["นาย", "นาง", "นางสาว", "ดร.", "พล.อ.", "พล.ต.อ."]
    for word, tag in combined_entities:
        w_clean = word.strip()
        if tag == "PERSON":
            for pfx in prefixes:
                if w_clean.startswith(pfx):
                    w_clean = w_clean[len(pfx):].strip()
        if len(w_clean) > 1:
            clean_entities.append((w_clean, tag))
            
    return clean_entities

# --- 3. Manage State ---
if "news_headline_state" not in st.session_state:
    st.session_state["news_headline_state"] = "“ทรงศักดิ์” ชี้แจงว่า “เอาอยู่” หมายถึงรัฐบาลเตรียมความพร้อม ไม่ได้หมายถึงเอาชนะธรรมชาติ"

if "news_body_state" not in st.session_state:
    st.session_state["news_body_state"] = """นายทรงศักดิ์ กล่าวด้วยว่า หากน้ำไม่สามารถระบายออกจากพื้นที่ได้ การสูบน้ำจากจุดหนึ่งไปยังอีกจุดหนึ่งก็อาจทำให้สูบน้ำกลับมาที่เดิม ดังนั้นโจทย์สำคัญคือการทำให้น้ำสามารถออกจากพื้นที่ชั้นในและระบายลงสู่ทะเลได้อย่างต่อเนื่อง

อย่างไรก็ตาม ปัจจัยสำคัญในขณะนี้คือปริมาณฝนที่ตกลงมาเกินกว่าที่คาดการณ์ ส่งผลให้ระบบระบายน้ำต้องรับภาระเพิ่มขึ้น ขณะที่คลองหลายแห่งมีน้ำเต็ม จึงจำเป็นต้องเร่งบริหารจัดการและสูบน้ำออกจากพื้นที่ชั้นในลงสู่ทะเล เพื่อเปิดพื้นที่ในระบบคลองให้สามารถรองรับน้ำและระบายต่อไปได้"""

# --- 4. Tabs ---
tab1, tab2 = st.tabs(["📰 1. วิเคราะห์ตัวเนื้อหาข่าว & แหล่งอ้างอิง (Content & Reference)", "💬 2. วิเคราะห์สัญญาณเตือนจากคอมเมนต์ (Comment Signals)"])

# ==========================================
# TAB 1: วิเคราะห์เนื้อหาข่าว + แหล่งอ้างอิง
# ==========================================
with tab1:
    st.subheader("1. ตรวจสอบลักษณะภาษา เนื้อหาข่าว และแหล่งข้อมูลอ้างอิง")
    
    col_input1, col_input2 = st.columns([1, 2])
    
    with col_input1:
        input_headline = st.text_input("พาดหัวข่าว (Headline):", value=st.session_state["news_headline_state"], key="headline_box")
    with col_input2:
        input_news_body = st.text_area("เนื้อหาข่าวแบบเต็ม (News Body):", value=st.session_state["news_body_state"], height=140, key="news_body_box")
        
    st.session_state["news_headline_state"] = input_headline
    st.session_state["news_body_state"] = input_news_body
    
    full_news_text = f"{input_headline}\n{input_news_body}"
    clickbait_words = ["เกิดมาเพิ่งเคยเห็น", "จมบาดาล", "แชร์ด่วน", "ช็อก", "หายขาด", "เตือนภัยด่วน", "ไม่อยากให้คุณรู้", "ก่อนโดนลบ", "ตะลึง", "อึ้ง"]

    if st.button("🔎 ตรวจสอบเนื้อหาและค้นหาแหล่งอ้างอิง", type="primary", key="btn_check_news"):
        if not full_news_text.strip():
            st.warning("กรุณาใส่เนื้อหาข่าวก่อนครับ")
        else:
            with st.spinner("⚡ AI กำลังประมวลผล ตรวจสอบโครงสร้างภาษา และค้นหาแหล่งอ้างอิง..."):
                col1, col2 = st.columns(2)
                
                sample_text = full_news_text[:400]
                candidate_labels = [
                    "รายงานข่าวข้อเท็จจริงการทำงานของรัฐหรือเอกชน (Official News)", 
                    "พาดหัวเกินจริงเพื่อเรียกร้องความสนใจ (Clickbait)", 
                    "ข่าวลือหรือข้อความไม่มีแหล่งอ้างอิง (Unverified Rumor)"
                ]
                
                res = classifier_pipe(sample_text, candidate_labels=candidate_labels)
                top_label = res['labels'][0]
                confidence = res['scores'][0] * 100
                found_clickbait = [w for w in clickbait_words if w in full_news_text]
                
                # --- คำนวณ Content Trust Score ---
                content_score = 90 if "Official News" in top_label else (40 if found_clickbait else 60)
                
                with col1:
                    st.markdown("### 🎯 ผลการประเมินประเภทข่าว")
                    if content_score >= 80:
                        st.success(f"**ลักษณะข่าว:** {top_label}")
                    else:
                        st.warning(f"**ลักษณะข่าว:** {top_label}")
                    st.caption(f"ความมั่นใจของ AI: {confidence:.2f}% | คะแนนความน่าเชื่อถือเชิงภาษา: {content_score}/100")

                with col2:
                    st.markdown("### ⚠️ สัญญาณความเสี่ยง Clickbait")
                    if found_clickbait:
                        st.warning(f"**ตรวจพบคำสุ่มเสี่ยง:** {', '.join(found_clickbait)}")
                    else:
                        st.success("ไม่พบคำเร้าอารมณ์สุ่มเสี่ยง ใช้ภาษารายงานข้อเท็จจริง")

                st.markdown("---")
                
                # --- ส่วนที่เพิ่มใหม่: เหตุผลของ AI & ลิงก์อ้างอิง ---
                st.markdown("### 🧠 ทำไม AI ถึงประเมินว่าข่าวนี้มีแนวโน้มเป็น 'ข่าวจริง/น่าเชื่อถือ'?")
                
                # ดึง NER หาบุคคล/สถานที่มาอ้างอิง
                clean_ner_input = full_news_text[:300].replace("\n", " ").strip()
                raw_ner_tags = ner_engine.tag(clean_ner_input)
                merged_entities = combine_ner_tokens(raw_ner_tags)
                entities_found = [w for w, t in merged_entities if t in ["PERSON", "LOCATION", "ORGANIZATION"]]
                
                col_reason, col_ref = st.columns(2)
                
                with col_reason:
                    st.markdown("**📌 เหตุผลสนับสนุนจาก AI (AI Reasoning):**")
                    reasons = []
                    if "Official News" in top_label or "ข้อเท็จจริง" in top_label:
                        reasons.append("✅ **มีโครงสร้างภาษารายงานเชิงนโยบาย/การทำงาน:** มีลักษณะเป็นการให้สัมภาษณ์หรือชี้แจงอย่างเป็นทางการ")
                    if len(entities_found) > 0:
                        reasons.append(f"✅ **มีการระบุบุคคล/องค์กรชัดเจน:** เช่น {', '.join(list(set(entities_found))[:3])} ซึ่งสามารถตรวจสอบยืนยันได้")
                    if not found_clickbait:
                        reasons.append("✅ **ปราศจากภาษาปั่นกระแส (No Clickbait):** ไม่พบคำกระตุ้นให้เกิดความตื่นตระหนกหรือบังคับแชร์")
                    
                    for r in reasons:
                        st.markdown(r)
                        
                with col_ref:
                    st.markdown("**🔗 ปุ่มตรวจสอบแหล่งอ้างอิงภายนอก (Cross-Verification):**")
                    st.caption("สามารถกดปุ่มด้านล่างเพื่อตรวจสอบข่าวนี้กับสำนักข่าวชั้นนำหรือ Google Search ได้ทันที:")
                    
                    # สร้าง Search URL สำหรับตรวจสอบข่าว
                    query = urllib.parse.quote(input_headline)
                    google_search_url = f"https://www.google.com/search?q={query}"
                    fact_check_url = f"https://www.google.com/search?q={urllib.parse.quote(input_headline + ' ข่าวจริง หรือ ข่าวปลอม')}"
                    
                    st.link_button("🌐 ค้นหาข่าวนี้ใน Google (เพื่อเทียบกับสำนักข่าวหลัก)", google_search_url, use_container_width=True)
                    st.link_button("🔎 ตรวจสอบประวัติการ Fact-Check ของข่าวนี้", fact_check_url, use_container_width=True)

                st.markdown("---")
                st.markdown("### 🏷️ ตารางองค์กร/บุคคล/สถานที่ ที่ถูกอ้างถึงในข่าว (Thai NER)")
                
                tag_map = {"PERSON": "ชื่อคน", "LOCATION": "สถานที่", "ORGANIZATION": "องค์กร"}
                extracted = [{"คำที่พบ (Entity)": w, "ประเภท": tag_map.get(t, t)} for w, t in merged_entities]
                if extracted:
                    st.dataframe(pd.DataFrame(extracted).drop_duplicates(), use_container_width=True)
                else:
                    st.info("ไม่พบการระบุชื่อบุคคล สถานที่ หรือองค์กรชัดเจน")

# ==========================================
# TAB 2: วิเคราะห์คอมเมนต์
# ==========================================
with tab2:
    st.subheader("2. วิเคราะห์สัญญาณหักล้าง/เตือนภัยจากคอมเมนต์ของผู้คน")
    
    current_news = f"พาดหัว: {st.session_state['news_headline_state']}\n\nเนื้อหา: {st.session_state['news_body_state']}"
    st.info(f"📌 **ข่าวที่กำลังอ้างอิงวิเคราะห์:**\n\n\"{current_news[:250]}...\"")
    
    default_comments = """รับทราบครับ เป็นกำลังใจให้เจ้าหน้าที่ทุกท่าน
นายทรงศักดิ์ลงพื้นที่เองเลย ขอบคุณครับ
ข่าวปลอมครับ เรื่องนี้ยังไม่มีการประชุมเลย
ภาพเก่าเอามาเล่าใหม่หรือเปล่าครับ"""

    comments_text = st.text_area("ก๊อปปี้คอมเมนต์มาวางที่นี่ (แยกบรรทัด):", value=default_comments, height=160, key="comments_input")

    if st.button("📊 ประมวลผลความน่าเชื่อถือจากคอมเมนต์", type="primary", key="btn_check_comments"):
        with st.spinner("⚡ กำลังวิเคราะห์ความคิดเห็น..."):
            comment_list = [c.strip() for c in comments_text.split("\n") if c.strip()]
            
            if not comment_list:
                st.warning("กรุณาใส่คอมเมนต์อย่างน้อย 1 บรรทัดครับ")
            else:
                fake_warning_words = ["ข่าวปลอม", "ปลอม", "เฟคนิวส์", "fake news", "ไม่จริง", "หลอกลวง", "อย่าเชื่อ", "ภาพเก่า", "บิดเบือน"]
                sarcasm_words = ["เอาอยู่", "ถอดบทเรียน", "ดีเยี่ยม", "เจริญ", "ทรงคุณค่า"]
                
                fake_signals = 0
                sarcasm_signals = 0
                neutral_signals = 0
                
                detailed_results = []
                
                for comment in comment_list:
                    is_fake_warning = any(w in comment.lower() for w in fake_warning_words)
                    is_sarcastic = any(w in comment for w in sarcasm_words)
                    
                    if is_fake_warning:
                        fake_signals += 1
                        status = "🔴 คอมเมนต์เตือนว่าเป็นข่าวปลอม/บิดเบือน"
                    elif is_sarcastic:
                        sarcasm_signals += 1
                        status = "🟠 คอมเมนต์ประชดประชัน/วิพากษ์วิจารณ์"
                    else:
                        neutral_signals += 1
                        status = "🟡 คอมเมนต์ทั่วไป/รายงานสถานการณ์"
                            
                    detailed_results.append({
                        "ความคิดเห็น": comment,
                        "ผลวิเคราะห์สัญญาณ": status
                    })
                
                total = len(comment_list)
                fake_percentage = (fake_signals / total) * 100
                
                st.markdown("---")
                st.markdown("### ⚖️ สรุปผลการตรวจสอบความน่าเชื่อถือ (Verdict)")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("เตือนว่าเป็น 'ข่าวปลอม/บิดเบือน'", f"{fake_signals} รายการ ({fake_percentage:.1f}%)")
                m2.metric("ประชดประชัน/วิจารณ์เชิงลบ", f"{sarcasm_signals} รายการ")
                m3.metric("รายงานสถานการณ์ทั่วไป", f"{neutral_signals} รายการ")
                
                st.markdown("#### 🛡️ ประเมินระดับความเสี่ยง (Risk Assessment)")
                if fake_percentage >= 30:
                    st.error("🚨 **ความเสี่ยงสูงมาก (High Risk of Fake News):** มีผู้คนในคอมเมนต์เตือนภัย/หักล้างข้อมูลอย่างชัดเจน!")
                elif sarcasm_signals >= 1:
                    st.warning("⚠️ **พบกระแสประชด/วิจารณ์เชิงลบ:** ข่าวนี้อาจมีประเด็นขัดแย้งเกี่ยวกับความถูกต้องในการจัดการสถานการณ์")
                else:
                    st.success("✅ **ยังไม่พบสัญญาณเตือนข่าวปลอมหลักจากคอมเมนต์**")

                st.markdown("#### ตารางแจกแจงคอมเมนต์รายบุคคล")
                st.dataframe(pd.DataFrame(detailed_results), use_container_width=True)
                