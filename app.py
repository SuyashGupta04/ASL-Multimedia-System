import streamlit as st
import os
import tempfile
import pandas as pd
import time
import cv2
import random
import string
from datetime import datetime

# --- LOAD ENGINES & UTILS ---
from engines.image_engine import ImageEngine
from engines.video_engine import VideoEngine, VideoDecoder
from engines.translator_engine import TranslatorEngine
from utils.auth import login_user, register_user
from utils.feedback import save_feedback, get_feedback_df

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ASL Storytelling System",
    layout="wide",
    page_icon="📖",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { 
        height: 50px; 
        white-space: pre-wrap; 
        background-color: #f0f2f6; 
        border-radius: 8px 8px 0px 0px; 
        padding: 10px; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: #ffffff; 
        border-top: 3px solid #6366f1;
        color: #6366f1;
    }
    .metric-card { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 10px; 
        text-align: center; 
        border: 1px solid #e0e0e0; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
    }
    .quiz-card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #eee;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #6366f1;
    }
</style>
""", unsafe_allow_html=True)

# --- STORYTELLING VOCABULARY & SENTENCES ---
COMMON_WORDS = [
    "HELLO", "PLEASE", "SORRY", "YES", "NO", "THANK",
    "HELP", "WATER", "MORE", "DONE", "EAT", "LOVE",
    "HAPPY", "SAD", "WORK", "FAMILY", "FRIEND", "SCHOOL"
]

STORY_SENTENCES = [
    "HELLO FRIEND HOW ARE YOU",
    "THE DOG IS HAPPY",
    "I LOVE MY FAMILY",
    "PLEASE HELP ME",
    "THE CAT IS SLEEPING",
    "GOOD MORNING TEACHER",
    "I WANT MORE WATER",
    "THANK YOU FOR HELPING",
    "THE BOOK IS BEAUTIFUL"
]

# --- SESSION STATE ---
DEFAULT_STATE = {
    'logged_in': False,
    'user_role': None,
    'user_name': None,
    'quiz_score': 0,
    'quiz_xp': 0,
    'quiz_level': 1,
    'quiz_lives': 3,
    'quiz_streak': 0,
    'quiz_current': None,
    'quiz_options': [],
    'recent_quiz_chars': [],
    'recent_quiz_words': [],
    'recent_quiz_stories': [],
    'last_quiz_type': "🔤 Alphabet",
    'quiz_story_video': None,
    'gen_video_path_t1': None,
    'gen_video_path_t2': None,
    'trans_out': None,
    'trans_text': None,
    'audio_path': None,
    'decode_result': None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- HIGH RANDOMNESS QUIZ HELPERS ---
def get_new_quiz_item(pool, memory_buffer, max_buffer_size=5):
    """Universal helper to fetch random items while preventing recent repeats."""
    available = [item for item in pool if item not in memory_buffer]
    if not available:
        memory_buffer.clear()
        available = pool
    new_item = random.choice(available)
    memory_buffer.append(new_item)
    if len(memory_buffer) > max_buffer_size:
        memory_buffer.pop(0)
    return new_item


# --- LOAD ENGINES ---
@st.cache_resource
def load_engines():
    try:
        return ImageEngine(), VideoEngine(), TranslatorEngine(), VideoDecoder()
    except Exception as e:
        st.error(f"Critical Error: Failed to load engines. {e}")
        return None, None, None, None


image_engine, video_engine, translator_engine, video_decoder = load_engines()

if not image_engine:
    st.stop()

# ==================================================
# 🔐 AUTHENTICATION
# ==================================================
if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🔐 ASL Storytelling System</h1>", unsafe_allow_html=True)
        tab_login, tab_reg = st.tabs(["Login", "Register"])

        with tab_login:
            with st.form("login_form"):
                user_in = st.text_input("Username")
                pass_in = st.text_input("Password", type="password")
                if st.form_submit_button("Log In", type="primary"):
                    user_data = login_user(user_in, pass_in)
                    if user_data:
                        st.session_state['logged_in'] = True
                        st.session_state['user_role'] = user_data['role']
                        st.session_state['user_name'] = user_data['name']
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password")

        with tab_reg:
            with st.form("reg_form"):
                new_user = st.text_input("New Username")
                new_name = st.text_input("Full Name")
                new_pass = st.text_input("New Password", type="password")
                if st.form_submit_button("Create Account"):
                    success, msg = register_user(new_user, new_pass, new_name)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    st.stop()

# ==================================================
# 🚀 MAIN APPLICATION
# ==================================================
with st.sidebar:
    st.title("📖 ASL Studio")
    st.write(f"**User:** {st.session_state['user_name']}")

    if st.session_state['user_role'] == 'admin':
        st.info("🛡️ Administrator Mode")
        mode_research = True
    else:
        st.success("👤 Learner Mode")
        c1, c2 = st.columns(2)
        c1.metric("🏆 Score", st.session_state['quiz_score'])
        c2.metric("🔥 Streak", st.session_state['quiz_streak'])
        mode_research = False

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.caption("v17.0 | Interactive Storytelling Edition")

if mode_research:
    tab_titles = ["🔤 Text-to-Sign", "🎬 Story Stitcher", "🔓 Video Decoder", "📹 Smart Translator", "🔬 Research Lab",
                  "📊 Feedback"]
else:
    tab_titles = ["🔤 Text-to-Sign", "🎬 Story Stitcher", "🔓 Video Decoder", "📹 Smart Translator",
                  "📖 Interactive Stories", "📝 Feedback"]

tabs = st.tabs(tab_titles)
tab1, tab2, tab3, tab4, tab5, tab6 = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4], tabs[5]

# ==================================================
# TAB 1: GENERATION (Finger-Spelling)
# ==================================================
with tab1:
    col_input, col_preview = st.columns([1, 2])
    with col_input:
        st.subheader("1. Input")
        text_input_a = st.text_input("Enter text:", placeholder="HELLO", key="input_a")
        speed_val = st.slider("Playback Speed", min_value=0.5, max_value=3.0, value=1.0, step=0.5)

        if st.button("🎬 Generate Video", type="primary", key="btn_anim_a"):
            if text_input_a:
                with st.status("🎬 Rendering Animation...", expanded=True) as status:
                    os.makedirs("temp_output", exist_ok=True)
                    path = f"temp_output/anim_a_{int(time.time())}.mp4"
                    # show_text=True is default, but explicitly written for clarity
                    res = video_engine.generate_sequence(text_input_a, path, force_spelling=True, speed=speed_val,
                                                         show_text=True)
                    if res:
                        status.update(label="✅ Video Ready!", state="complete", expanded=False)
                        st.session_state['gen_video_path_t1'] = res
                        st.rerun()
                    else:
                        status.update(label="❌ Generation Failed", state="error")

    with col_preview:
        st.subheader("2. Result")
        if st.session_state['gen_video_path_t1'] and os.path.exists(st.session_state['gen_video_path_t1']):
            st.video(st.session_state['gen_video_path_t1'])
            with open(st.session_state['gen_video_path_t1'], "rb") as v_file:
                st.download_button("⬇️ Download Video", v_file, "asl_fingerspelling.mp4", mime="video/mp4")

# ==================================================
# TAB 2: SMART STITCHER
# ==================================================
with tab2:
    st.markdown("#### 🎬 Story Stitcher Engine")
    col_st_in, col_st_out = st.columns([1, 2])
    with col_st_in:
        text_input_b = st.text_input("Enter sentence:", placeholder="HOW ARE YOU", key="input_b")
        speed_val_b = st.slider("Playback Speed", min_value=0.5, max_value=3.0, value=1.0, step=0.5, key="speed_b")

        if st.button("🎬 Generate Smart Video", type="primary", key="btn_stitch"):
            if text_input_b:
                with st.status("🏗️ Building Video Sequence...", expanded=True) as status:
                    os.makedirs("temp_output", exist_ok=True)
                    path = f"temp_output/smart_stitch_{int(time.time())}.mp4"
                    # show_text=True is default, but explicitly written for clarity
                    res = video_engine.generate_sequence(text_input_b, path, force_spelling=False, speed=speed_val_b,
                                                         show_text=True)
                    if res:
                        status.update(label="✅ Video Ready!", state="complete", expanded=False)
                        st.session_state['gen_video_path_t2'] = res
                        st.rerun()
                    else:
                        status.update(label="❌ Generation Failed", state="error")

    with col_st_out:
        if st.session_state['gen_video_path_t2'] and os.path.exists(st.session_state['gen_video_path_t2']):
            st.video(st.session_state['gen_video_path_t2'])
            with open(st.session_state['gen_video_path_t2'], "rb") as v_file:
                st.download_button("⬇️ Download Video", v_file, "asl_smart_stitch.mp4", mime="video/mp4")

# ==================================================
# TAB 3: MODE 3 (VISIBLE WORDS ONLY)
# ==================================================
with tab3:
    st.markdown("#### 🔓 Video Text Decoder")
    uploaded_decode = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], key="decoder_words")

    if uploaded_decode is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_decode.read())
        video_path = tfile.name

        if st.button("🔍 Extract Visible Text", type="primary"):
            with st.spinner("Analyzing frames via OCR..."):
                result = video_decoder.decode_mode2_visible(video_path)
                st.divider()
                st.subheader("Result")
                if result and "Error" not in str(result):
                    st.success("✅ Extracted Text:")
                    st.info(result)
                else:
                    st.error(result)
        os.remove(video_path)

# ==================================================
# TAB 4: TRANSLATOR (ADVANCED)
# ==================================================
with tab4:
    st.markdown("#### 📹 Smart Translator & AI Decoder")

    c_mode, c_detect, c_voice = st.columns([2, 2, 1])
    with c_mode:
        mode_selection = st.radio("Input Source", ["🔴 Live Webcam", "📂 Upload Video"], horizontal=True,
                                  label_visibility="collapsed")
    with c_detect:
        detect_mode = st.selectbox("Target", ["Letter (A-Z)", "Word (Common Signs)"], label_visibility="collapsed")
    with c_voice:
        enable_tts = st.toggle("🔊 Voice", value=True)
        show_trace = st.toggle("✨ Motion", value=True)

    if "Webcam" in mode_selection:
        run_cam = st.toggle("Start Camera", key="cam_toggle")
        col_cam, col_stats = st.columns([2, 1])
        with col_cam:
            frame_placeholder = st.empty()
        with col_stats:
            st.markdown("### Status")
            txt_display = st.empty()
            conf_bar = st.progress(0)
            conf_label = st.empty()
            st.info(f"Scanning for: **{detect_mode}**")

        if run_cam:
            cap = cv2.VideoCapture(0)
            last_spoken = ""
            while cap.isOpened() and run_cam:
                ret, frame = cap.read()
                if not ret: break

                engine_mode = "Letter" if "Letter" in detect_mode else "Word"
                proc_frame, text, confidence = translator_engine.process_frame(
                    frame, detection_mode=engine_mode, draw_trace=show_trace
                )

                frame_placeholder.image(proc_frame, channels="BGR")
                conf_bar.progress(float(confidence))
                conf_label.caption(f"Certainty: {int(confidence * 100)}%")

                if text:
                    txt_display.markdown(f"# 🟢 {text}")
                    if enable_tts and confidence > 0.8 and text != last_spoken:
                        st.toast(f"🗣️ Speaking: {text}", icon="🔊")
                        last_spoken = text
                else:
                    txt_display.markdown(f"# 🔴 ...")
                time.sleep(0.03)
            cap.release()
    else:
        st.info(f"Frame-by-frame analysis for **{detect_mode}**.")
        up_file = st.file_uploader("Upload Video", type=["mp4"])
        if up_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(up_file.read())
            if st.button("🚀 Analyze Video", type="primary"):
                with st.spinner("Processing..."):
                    out_video, final_text = translator_engine.process_video_smart(tfile.name)
                    st.session_state['trans_out'] = out_video
                    st.session_state['trans_text'] = final_text
            if st.session_state.get('trans_text'):
                st.success(f"**Detected:** {st.session_state['trans_text']}")
                st.video(st.session_state['trans_out'])

# ==================================================
# TAB 5: RESEARCH & INTERACTIVE STORIES LAB
# ==================================================
with tab5:
    if mode_research:
        # --- ADMIN RESEARCH LAB ---
        st.markdown("## 🧪 Computer Vision Research Laboratory")
        st.caption("Experiment: Comparative Analysis of Feature Extraction Algorithms for ASL")
        c_setup, c_realtime = st.columns([1, 2])
        with c_setup:
            bench_file = st.file_uploader("📂 Upload Benchmark Dataset (Video)", type=["mp4"])
            inject_noise = st.checkbox("Simulate Noise (Gaussian $\sigma=25$)")
            run_exp = st.button("🚀 Run Experiment", type="primary")
        with c_realtime:
            if run_exp and bench_file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(bench_file.read())
                with st.spinner("🔄 Processing frames..."):
                    raw_data = translator_engine.run_research_benchmark(tfile.name, "assets/images", inject_noise)
                if raw_data:
                    df_res = pd.DataFrame({"Frame": range(len(raw_data['mse_time'])), "MSE": raw_data['mse_time'],
                                           "NCC": raw_data['ncc_time'], "ORB": raw_data['orb_time']})
                    st.line_chart(df_res.set_index("Frame"), width="stretch")
    else:
        # --- INTERACTIVE STORIES (GAMIFIED) ---
        c_lvl, c_xp, c_life = st.columns([1, 3, 1])
        with c_lvl:
            st.metric("Level", f"{st.session_state['quiz_level']}")
        with c_xp:
            xp_needed = st.session_state['quiz_level'] * 100
            cur_xp = st.session_state['quiz_xp']
            st.write(f"**XP: {cur_xp} / {xp_needed}**")
            st.progress(min(1.0, cur_xp / xp_needed) if xp_needed > 0 else 0)
        with c_life:
            st.markdown(f"### {'❤️' * st.session_state['quiz_lives']}")
            if st.session_state['quiz_lives'] == 0:
                st.error("JOURNEY OVER!")
                if st.button("🔄 Restart Journey"):
                    st.session_state.update(
                        {'quiz_lives': 3, 'quiz_score': 0, 'quiz_xp': 0, 'quiz_level': 1, 'recent_quiz_chars': [],
                         'recent_quiz_words': [], 'recent_quiz_stories': []})
                    st.rerun()
                st.stop()

        st.divider()

        # Learning Path Selector
        c_qtype, c_ansmode = st.columns([2, 1])
        with c_qtype:
            quiz_type = st.radio("Learning Path:", ["🔤 Alphabet", "💬 Vocabulary", "📖 Story Decoder"], horizontal=True)
        with c_ansmode:
            # Disable Mimic for stories (too complex for webcam processing)
            quiz_mode = st.radio("Answer Mode:", ["🅰️ Multiple Choice", "📹 Mimic Master"], horizontal=True,
                                 disabled=(quiz_type == "📖 Story Decoder"))

        # Reset Question if Type Changes
        if quiz_type != st.session_state['last_quiz_type']:
            st.session_state['last_quiz_type'] = quiz_type
            st.session_state['quiz_current'] = None
            st.session_state['quiz_story_video'] = None

        # --- QUESTION GENERATION ---
        if st.session_state['quiz_current'] is None:
            if quiz_type == "🔤 Alphabet":
                max_idx = min(26, st.session_state['quiz_level'] * 5)
                pool = list(string.ascii_uppercase[:max_idx])
                target = get_new_quiz_item(pool, st.session_state['recent_quiz_chars'], max_buffer_size=10)
            elif quiz_type == "💬 Vocabulary":
                pool = COMMON_WORDS
                target = get_new_quiz_item(pool, st.session_state['recent_quiz_words'], max_buffer_size=5)
            else:  # Story Decoder
                pool = STORY_SENTENCES
                target = get_new_quiz_item(pool, st.session_state['recent_quiz_stories'], max_buffer_size=4)

            st.session_state['quiz_current'] = target

            # Generate False Options
            opts = [target]
            while len(opts) < 4:
                c = random.choice(pool)
                if c not in opts: opts.append(c)
            random.shuffle(opts)
            st.session_state['quiz_options'] = opts

            # Pre-generate the story video
            if quiz_type == "📖 Story Decoder":
                with st.spinner("🎬 Animating ASL Story..."):
                    ts = int(time.time())
                    temp_story_path = os.path.join("temp_output", f"quiz_story_{ts}.mp4")
                    os.makedirs("temp_output", exist_ok=True)
                    # --- FIX: hide_text is forced to False so it doesn't reveal the answer! ---
                    video_engine.generate_sequence(target, temp_story_path, force_spelling=False, speed=1.0,
                                                   show_text=False)
                    st.session_state['quiz_story_video'] = temp_story_path

        target = st.session_state['quiz_current']
        col_q, col_a = st.columns([1, 1])

        # --- DISPLAY QUESTION ---
        # --- DISPLAY QUESTION ---
        with col_q:
            if quiz_type == "📖 Story Decoder":
                st.info("Translate the story being told:")
                if st.session_state['quiz_story_video'] and os.path.exists(st.session_state['quiz_story_video']):
                    # FIX: Read as bytes to prevent Streamlit MediaFileStorageError
                    with open(st.session_state['quiz_story_video'], 'rb') as v_file:
                        st.video(v_file.read(), format="video/mp4", autoplay=True, muted=True, loop=True)
                else:
                    st.error("Failed to render story.")
            else:
                st.info("What does this sign mean?")
                path = video_engine.get_sign_asset(
                    target) if quiz_type == "🔤 Alphabet" else video_engine.fetch_online_sign(target)

                if path and os.path.exists(path):
                    if path.endswith(".mp4"):
                        # FIX: Read video as bytes
                        with open(path, 'rb') as v_file:
                            st.video(v_file.read(), format="video/mp4", autoplay=True, muted=True, loop=True)
                    else:
                        # FIX: Read image as bytes
                        with open(path, 'rb') as img_file:
                            st.image(img_file.read(), width=300)
                else:
                    st.warning(f"Missing asset for '{target.upper()}'")

        # --- ANSWER LOGIC ---
        with col_a:
            if "Multiple" in quiz_mode or quiz_type == "📖 Story Decoder":
                def check_mcq(sel):
                    if sel == target:
                        xp_gain = 30 if quiz_type == "📖 Story Decoder" else (15 if quiz_type == "💬 Vocabulary" else 10)
                        st.session_state['quiz_xp'] += xp_gain
                        st.session_state['quiz_score'] += xp_gain
                        st.session_state['quiz_streak'] += 1
                        st.toast("✅ Excellent Translation!", icon="🌟")

                        if st.session_state['quiz_xp'] >= st.session_state['quiz_level'] * 100:
                            st.session_state['quiz_level'] += 1
                            st.balloons()
                    else:
                        st.session_state['quiz_lives'] -= 1
                        st.session_state['quiz_streak'] = 0
                        st.toast(f"❌ Not quite. The correct answer was: {target}", icon="💔")

                    st.session_state['quiz_current'] = None
                    st.session_state['quiz_story_video'] = None
                    time.sleep(1)


                ops = st.session_state['quiz_options']
                st.markdown("### Select the correct translation:")

                # Stack buttons vertically for stories so long text fits
                if quiz_type == "📖 Story Decoder":
                    for i in range(4):
                        if st.button(ops[i].upper(), key=f"btn_{i}", use_container_width=True):
                            check_mcq(ops[i])
                            st.rerun()
                else:
                    b1, b2 = st.columns(2)
                    b3, b4 = st.columns(2)
                    if b1.button(ops[0].upper(), use_container_width=True): check_mcq(ops[0]); st.rerun()
                    if b2.button(ops[1].upper(), use_container_width=True): check_mcq(ops[1]); st.rerun()
                    if b3.button(ops[2].upper(), use_container_width=True): check_mcq(ops[2]); st.rerun()
                    if b4.button(ops[3].upper(), use_container_width=True): check_mcq(ops[3]); st.rerun()

            else:
                st.markdown(f"### Sign: **{target.upper()}**")

                if quiz_type == "💬 Vocabulary":
                    st.info("💡 Make sure your ML Model supports Word Level detection to play Mimic Master with words!")

                run_mimic = st.toggle("Start Cam")
                cam_ph = st.empty()
                if run_mimic:
                    cap = cv2.VideoCapture(0)
                    cnt = 0
                    while cap.isOpened() and run_mimic:
                        ret, frame = cap.read()
                        if not ret: break

                        engine_detect_mode = "Word" if quiz_type == "💬 Vocabulary" else "Letter"
                        frame, det, _ = translator_engine.process_frame(frame, engine_detect_mode, draw_trace=False)
                        cam_ph.image(frame, channels="BGR")

                        if det and det.upper() == target.upper():
                            cnt += 1
                            if cnt > 10:
                                st.session_state['quiz_xp'] += 25 if quiz_type == "💬 Vocabulary" else 20
                                st.session_state['quiz_score'] += 20
                                st.balloons()
                                if st.session_state['quiz_xp'] >= st.session_state['quiz_level'] * 100:
                                    st.session_state['quiz_level'] += 1
                                st.session_state['quiz_current'] = None
                                cap.release()
                                time.sleep(1)
                                st.rerun()
                        else:
                            cnt = 0
                        time.sleep(0.03)
                    cap.release()

        st.divider()
        if st.button("⏭️ Skip Question (-5 XP)"):
            st.session_state['quiz_xp'] = max(0, st.session_state['quiz_xp'] - 5)
            st.session_state['quiz_current'] = None
            st.session_state['quiz_story_video'] = None
            st.rerun()

# ==================================================
# TAB 6: FEEDBACK
# ==================================================
with tab6:
    st.markdown("### 📊 UX Lab")
    if mode_research:
        df = get_feedback_df()
        if not df.empty:
            k1, k2, k3 = st.columns(3)
            k1.metric("Responses", len(df))
            k2.metric("Avg Rating", f"{df['rating'].mean():.1f}")
            k3.metric("Avg SUS", f"{df['sus_score'].mean():.1f}")
            st.dataframe(df, width="stretch")
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "ux_data.csv")
    else:
        with st.form("research_survey"):
            c1, c2 = st.columns(2)
            with c1:
                r = st.slider("Rating", 1, 5, 5)
                cat = st.selectbox("Category", ["Accuracy", "Speed", "UI", "Other"])
            with c2:
                q1 = st.slider("Easy to use?", 1, 5, 4)
                q2 = st.slider("Confident?", 1, 5, 4)
            cm = st.text_area("Comments")
            sus = ((q1 - 1) + (q2 - 1)) * 12.5
            if st.form_submit_button("Submit"):
                save_feedback(st.session_state['user_name'], r, cat, sus, cm)
                st.success("Feedback Saved!")