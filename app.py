import streamlit as st
import os
import tempfile
import pandas as pd
import time
import cv2
import random
from datetime import datetime

# --- LOAD ENGINES & UTILS ---
from engines.image_engine import ImageEngine
from engines.video_engine import VideoEngine, VideoDecoder
from engines.translator_engine import TranslatorEngine
from utils.auth import login_user, register_user
from utils.feedback import save_feedback, get_feedback_df

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ASL Multimedia System",
    layout="wide",
    page_icon="🎓",
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
        border-top: 3px solid #4CAF50;
        color: #4CAF50;
    }
    .metric-card { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 10px; 
        text-align: center; 
        border: 1px solid #e0e0e0; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
DEFAULT_STATE = {
    'logged_in': False,
    'user_role': None,
    'user_name': None,
    'quiz_score': 0,
    'quiz_current': None,
    'quiz_options': [],
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
        st.markdown("<h1 style='text-align: center;'>🔐 ASL Multimedia System</h1>", unsafe_allow_html=True)
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
    st.title("🎓 ASL System")
    st.write(f"**User:** {st.session_state['user_name']}")

    if st.session_state['user_role'] == 'admin':
        st.info("🛡️ Administrator Mode")
        mode_research = True
    else:
        st.success("👤 Standard User Mode")
        st.markdown(f"**🏆 Quiz Score:** `{st.session_state['quiz_score']}`")
        mode_research = False

    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.caption("v8.8 | Graphical Research Lab")

# --- TABS CONFIGURATION ---
if mode_research:
    tab_titles = ["🔤 Text-to-Sign", "🎬 Smart Stitcher", "🔓 Mode 3: Words", "📹 Translator / Decoder", "🔬 Research Lab",
                  "📊 Feedback"]
else:
    tab_titles = ["🔤 Text-to-Sign", "🎬 Smart Stitcher", "🔓 Mode 3: Words", "📹 Translator / Decoder", "📚 Learn ASL",
                  "📝 Feedback"]

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
        if st.button("🎬 Generate Video", type="primary", key="btn_anim_a"):
            if text_input_a:
                with st.status("🎬 Rendering Animation...", expanded=True) as status:
                    os.makedirs("temp_output", exist_ok=True)
                    path = "temp_output/anim_a.mp4"
                    res = video_engine.generate_sequence(text_input_a, path, force_spelling=True)
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
    st.markdown("#### 🎬 Smart Text-to-Video Engine")
    col_st_in, col_st_out = st.columns([1, 2])
    with col_st_in:
        text_input_b = st.text_input("Enter sentence:", placeholder="HOW ARE YOU", key="input_b")
        if st.button("🎬 Generate Smart Video", type="primary", key="btn_stitch"):
            if text_input_b:
                with st.status("🏗️ Building Video Sequence...", expanded=True) as status:
                    os.makedirs("temp_output", exist_ok=True)
                    path = "temp_output/smart_stitch.mp4"
                    res = video_engine.generate_sequence(text_input_b, path, force_spelling=False)
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
    st.markdown("#### 🔓 Mode 3: Visible Text Decoder (Words)")
    st.caption("Extracts visible text slides (Mode 2) from the end of the video.")

    uploaded_decode = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], key="decoder_words")

    if uploaded_decode is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_decode.read())
        video_path = tfile.name

        if st.button("🔍 Extract Visible Text", type="primary"):
            with st.spinner("Analyzing frames via OCR..."):
                # MODE 2: VISIBLE WORDS
                result = video_decoder.decode_mode2_visible(video_path)

                st.divider()
                st.subheader("Result")
                if result and "Error" not in str(result):
                    st.success("✅ Extracted Text:")
                    st.info(result)
                else:
                    st.error(result)
                    if "pytesseract" in str(result):
                        st.warning("Install Tesseract-OCR to use this feature.")

        os.remove(video_path)

# ==================================================
# TAB 4: TRANSLATOR / HIDDEN DECODER
# ==================================================
with tab4:
    st.markdown("#### 📹 Smart Translator & Hidden Decoder")

    mode_selection = st.radio("Select Input Mode:",
                              ["🔴 Live Webcam (Sign Translation)",
                               "📂 Upload Video (Frame-by-Frame Analysis)"],
                              horizontal=True)

    # --- OPTION 1: LIVE WEBCAM ---
    if "Webcam" in mode_selection:
        st.info("Translates sign language gestures into text in real-time.")
        run_cam = st.toggle("Start Camera")
        frame_placeholder = st.empty()

        if run_cam:
            cap = cv2.VideoCapture(0)
            while cap.isOpened() and run_cam:
                ret, frame = cap.read()
                if not ret: break

                proc, text = translator_engine.process_frame(frame)

                frame_placeholder.image(proc, channels="RGB")
                time.sleep(0.03)
            cap.release()

    # --- OPTION 2: UPLOAD VIDEO ---
    else:
        st.info("Analyzes uploaded video frame-by-frame to detect Signs.")
        up_file = st.file_uploader("Upload Video", type=["mp4"], key="trans_up")

        if up_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(up_file.read())

            if st.button("🚀 Analyze Video", type="primary"):
                with st.spinner("Analyzing frames..."):
                    out_video, final_text = translator_engine.process_video_smart(tfile.name)
                    st.session_state['trans_out'] = out_video
                    st.session_state['trans_text'] = final_text

            if st.session_state.get('trans_text'):
                st.success(f"**Detected:** {st.session_state['trans_text']}")
                st.video(st.session_state['trans_out'])

# ==================================================
# TAB 5: RESEARCH OR LEARN (GRAPHICAL UPDATES HERE)
# ==================================================
with tab5:
    if mode_research:
        st.subheader("🔬 Research Benchmarking Lab")
        st.caption("Compare computer vision algorithms for latency and stability.")

        bench_file = st.file_uploader("Upload Benchmark Video", type=["mp4"])
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            inject_noise = st.checkbox("Inject Gaussian Noise")

        if bench_file and st.button("📊 Run Graphical Analysis", type="primary"):
            bfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            bfile.write(bench_file.read())

            # 1. Run Benchmark Engine
            with st.spinner("Benchmarking Algorithms (MSE vs NCC vs ORB)..."):
                data = translator_engine.run_research_benchmark(bfile.name, "assets/images", inject_noise)

            if data:
                # 2. Process Data for Graphs
                avg_mse = sum(data['mse_time']) / len(data['mse_time'])
                avg_ncc = sum(data['ncc_time']) / len(data['ncc_time'])
                avg_orb = sum(data['orb_time']) / len(data['orb_time'])

                st.divider()
                st.markdown("### 📈 Performance Dashboard")

                # --- METRICS ROW ---
                m1, m2, m3 = st.columns(3)
                m1.metric("MSE (Pixel)", f"{avg_mse:.2f} ms", delta="Fastest", delta_color="normal")
                m2.metric("NCC (Template)", f"{avg_ncc:.2f} ms", delta="Medium", delta_color="off")
                m3.metric("ORB (Feature)", f"{avg_orb:.2f} ms", delta="-Slower", delta_color="inverse")

                # --- GRAPH 1: BAR CHART (AVERAGE LATENCY) ---
                st.subheader("1. Average Processing Speed (Lower is Better)")
                df_avg = pd.DataFrame({
                    "Algorithm": ["MSE", "NCC", "ORB"],
                    "Latency (ms)": [avg_mse, avg_ncc, avg_orb]
                })
                st.bar_chart(df_avg.set_index("Algorithm"), color="#4CAF50")

                # --- GRAPH 2: LINE CHART (FRAME-BY-FRAME STABILITY) ---
                st.subheader("2. Real-time Stability (Frame-by-Frame)")
                st.caption("Lower spikes indicate better stability.")
                df_frames = pd.DataFrame({
                    "MSE": data['mse_time'],
                    "NCC": data['ncc_time'],
                    "ORB": data['orb_time']
                })
                st.line_chart(df_frames)

                # --- GRAPH 3: SIMULATED ROBUSTNESS (STATIC) ---
                st.subheader("3. Noise Robustness (Accuracy Impact)")
                st.caption("Simulated data showing accuracy drop under noise.")
                df_robust = pd.DataFrame({
                    "Algorithm": ["MSE", "NCC", "ORB"],
                    "Accuracy Drop (%)": [45, 25, 5]
                })
                st.bar_chart(df_robust.set_index("Algorithm"), color="#FF4B4B")

            else:
                st.error("Benchmark failed. Try a longer video.")

    else:
        st.subheader("🧠 ASL Quiz")
        if st.session_state['quiz_current'] is None:
            char = random.choice("abcdefghijklmnopqrstuvwxyz")
            st.session_state['quiz_current'] = char
            opts = [char]
            while len(opts) < 4:
                c = random.choice("abcdefghijklmnopqrstuvwxyz")
                if c not in opts: opts.append(c)
            random.shuffle(opts)
            st.session_state['quiz_options'] = opts

        target = st.session_state['quiz_current']
        path = video_engine.get_image_path(target)
        if path: st.image(path, width=200)

        with st.form("quiz"):
            ans = st.radio("Answer:", [o.upper() for o in st.session_state['quiz_options']])
            if st.form_submit_button("Submit"):
                if ans.lower() == target:
                    st.success("Correct!")
                    st.session_state['quiz_score'] += 10
                    st.session_state['quiz_current'] = None
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Wrong.")

# ==================================================
# TAB 6: FEEDBACK
# ==================================================
with tab6:
    if mode_research:
        df = get_feedback_df()
        if not df.empty:
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df['rating'].value_counts())
    else:
        with st.form("feed_form", clear_on_submit=True):
            r = st.slider("Rating", 1, 5, 5)
            c = st.text_area("Comments")
            if st.form_submit_button("Submit"):
                save_feedback(st.session_state['user_name'], r, c)
                st.toast("Feedback Saved!")