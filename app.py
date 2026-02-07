import streamlit as st
import os
import tempfile
import pandas as pd
import time
import subprocess
import cv2
import random
import matplotlib.pyplot as plt
from datetime import datetime

# --- LOAD ENGINES & UTILS ---
from engines.image_engine import ImageEngine
from engines.video_engine import VideoEngine
from engines.translator_engine import TranslatorEngine
from utils.auth import login_user, register_user
from utils.feedback import save_feedback, get_feedback_df  # NEW IMPORT

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
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #4CAF50; }
    .auth-container { padding: 20px; border-radius: 10px; background-color: #f9f9f9; border: 1px solid #ddd; }
    .metric-card { background-color: #ffffff; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #eee; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_role' not in st.session_state: st.session_state['user_role'] = None
if 'user_name' not in st.session_state: st.session_state['user_name'] = None
if 'quiz_score' not in st.session_state: st.session_state['quiz_score'] = 0
if 'quiz_current' not in st.session_state: st.session_state['quiz_current'] = None
if 'gen_video_path_t1' not in st.session_state: st.session_state['gen_video_path_t1'] = None
if 'gen_video_path_t2' not in st.session_state: st.session_state['gen_video_path_t2'] = None


# --- LOAD ENGINES ---
@st.cache_resource
def load_engines():
    return ImageEngine(), VideoEngine(), TranslatorEngine()


image_engine, video_engine, translator_engine = load_engines()

# ==================================================
# 🔐 AUTHENTICATION SCREEN
# ==================================================
if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 ASL Multimedia System")
        st.markdown("Please log in to access the system.")

        tab_login, tab_reg = st.tabs(["Login", "Register"])

        # --- LOGIN TAB ---
        with tab_login:
            with st.form("login_form"):
                user_in = st.text_input("Username")
                pass_in = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Log In", type="primary")

                if submit_login:
                    user_data = login_user(user_in, pass_in)
                    if user_data:
                        st.session_state['logged_in'] = True
                        st.session_state['user_role'] = user_data['role']
                        st.session_state['user_name'] = user_data['name']
                        st.success(f"Welcome back, {user_data['name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password")

        # --- REGISTER TAB ---
        with tab_reg:
            with st.form("reg_form"):
                new_user = st.text_input("New Username")
                new_name = st.text_input("Full Name")
                new_pass = st.text_input("New Password", type="password")
                submit_reg = st.form_submit_button("Create Account")

                if submit_reg:
                    if new_user and new_pass:
                        success, msg = register_user(new_user, new_pass, new_name)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please fill all fields.")

    st.stop()  # Stop here if not logged in

# ==================================================
# 🚀 MAIN APPLICATION (LOGGED IN)
# ==================================================

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎓 ASL System")
    st.write(f"👤 **User:** {st.session_state['user_name']}")

    if st.session_state['user_role'] == 'admin':
        st.markdown("🛡️ **Role:** Administrator")
    else:
        st.markdown("👤 **Role:** Standard User")
        st.markdown(f"🏆 **Quiz Score:** {st.session_state['quiz_score']}")

    st.markdown("---")
    if st.button("🚪 Logout"):
        st.session_state['logged_in'] = False
        st.session_state['user_role'] = None
        st.session_state['quiz_score'] = 0
        st.session_state['gen_video_path_t1'] = None
        st.session_state['gen_video_path_t2'] = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 📂 System Status")

    if os.path.exists("assets/images"):
        st.success("✅ Images: Ready")
    else:
        st.error("❌ Images: Missing")

    if os.path.exists("assets/video_cache"):
        st.success("✅ Video Cache: Ready")
    else:
        st.warning("⚠️ Cache Empty")

    st.caption("v8.0 | Feedback Edition")

# --- MAIN TITLE ---
st.title("🗣️ Multimedia ASL Storytelling System")

# --- ROLE BASED TABS ---
# 5 TABS NOW
if st.session_state['user_role'] == 'admin':
    tabs = st.tabs(["🔤 Text-to-Sign", "🎬 Smart Stitcher", "📹 Smart Translator", "🔬 Research Lab", "📊 Admin Feedback"])
    mode_research = True
else:
    tabs = st.tabs(["🔤 Text-to-Sign", "🎬 Smart Stitcher", "📹 Smart Translator", "📚 Learn ASL", "📝 Feedback"])
    mode_research = False

tab1, tab2, tab3, tab4, tab5 = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]

# ==================================================
# TAB 1: GENERATION (Static + Finger-Spelling Video)
# ==================================================
with tab1:
    col_input, col_preview = st.columns([1, 2])
    with col_input:
        st.subheader("1. Input")
        text_input_a = st.text_input("Enter text:", placeholder="HELLO", key="input_a")
        st.markdown("### Actions")
        btn_ppt = st.button("📄 Generate PPT", key="btn_ppt_a")
        btn_anim = st.button("🎬 Generate Video", type="primary", key="btn_anim_a")

    with col_preview:
        st.subheader("2. Result")
        if text_input_a:
            words = text_input_a.split()
            # 1. Static Strip
            with st.expander("👁️ View Static Strip", expanded=True):
                for w in words:
                    strip = image_engine.create_word_strip(w)
                    if strip is not None:
                        st.image(strip, caption=w.upper(), use_container_width=True)

            # 2. Generate PPT
            if btn_ppt:
                with st.spinner("Creating PowerPoint..."):
                    os.makedirs("temp_output", exist_ok=True)
                    path = "temp_output/story_a.pptx"
                    image_engine.generate_ppt(text_input_a, path)
                    with open(path, "rb") as f:
                        st.download_button("⬇️ Download PPT", f, "story_a.pptx")
                        st.success("PPT Generated!")

            # 3. Generate Video
            if btn_anim:
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
                        st.error("Failed to generate video.")
        else:
            st.info("Enter text to see visualization.")

        if st.session_state['gen_video_path_t1'] and os.path.exists(st.session_state['gen_video_path_t1']):
            st.success("✅ **Video Output (Finger-Spelling)**")
            st.video(st.session_state['gen_video_path_t1'])

            with open(st.session_state['gen_video_path_t1'], "rb") as v_file:
                st.download_button(
                    label="⬇️ Download Video",
                    data=v_file,
                    file_name="asl_fingerspelling.mp4",
                    mime="video/mp4",
                    key="dl_t1"
                )

# ==================================================
# TAB 2: SMART STITCHER (Text -> Cache -> Web -> Image)
# ==================================================
with tab2:
    st.markdown("#### 🎬 Smart Text-to-Video Engine")
    st.markdown("Convert full sentences into video. Logic: **Cache** ➔ **Web (SignASL)** ➔ **Finger-Spelling**.")

    col_st_in, col_st_out = st.columns([1, 2])

    with col_st_in:
        text_input_b = st.text_input("Enter full sentence:", placeholder="HELLO I AM SUYASH", key="input_b")
        if st.button("🎬 Generate Video", type="primary", key="btn_stitch"):
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
                        st.error("Could not generate video.")

    with col_st_out:
        if st.session_state['gen_video_path_t2'] and os.path.exists(st.session_state['gen_video_path_t2']):
            st.success("✅ **Video Output (Smart)**")
            st.video(st.session_state['gen_video_path_t2'])

            with open(st.session_state['gen_video_path_t2'], "rb") as v_file:
                st.download_button(
                    label="⬇️ Download Video",
                    data=v_file,
                    file_name="asl_smart_stitch.mp4",
                    mime="video/mp4",
                    key="dl_t2"
                )

# ==================================================
# TAB 3: SMART TRANSLATOR (HYBRID)
# ==================================================
with tab3:
    st.markdown("#### 📹 Universal Sign Decoder")

    col_sets, col_up = st.columns([1, 2])
    with col_sets:
        st.subheader("Input Source")
        mode = st.radio("Select Source:", ["🔴 Live Webcam", "📂 Upload Video"])

        st.info(
            "ℹ️ **AI Logic:** \n1. Checks for static letters (A-Z).\n2. If not found, checks for human skeleton.\n3. Outputs text or [SIGN] tag.")

    with col_up:
        # LIVE WEBCAM
        if mode == "🔴 Live Webcam":
            st.info("Uses your webcam directly in the browser.")
            run_cam = st.checkbox("Start Webcam Stream")

            if run_cam:
                frame_placeholder = st.empty()
                status_placeholder = st.empty()
                cap = cv2.VideoCapture(0)

                while cap.isOpened() and run_cam:
                    ret, frame = cap.read()
                    if not ret: break

                    processed_frame, detected_text = translator_engine.process_frame(frame)
                    frame_placeholder.image(processed_frame, channels="RGB")
                    status_placeholder.info(f"Detected: **{detected_text}**")

                cap.release()
            else:
                st.write("Click checkbox to start.")

        # UPLOAD VIDEO
        else:
            up_file = st.file_uploader("Upload .mp4 Video", type=["mp4"])
            if up_file:
                col_vid_in, col_vid_out = st.columns(2)
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(up_file.read())

                with col_vid_in:
                    st.markdown("**Input Video**")
                    st.video(tfile.name)

                    if st.button("🚀 Run Smart Translation", type="primary"):
                        with st.spinner("Running Hybrid Analysis..."):
                            out_video, final_text = translator_engine.process_video_smart(tfile.name)

                            st.session_state['trans_out'] = out_video
                            st.session_state['trans_text'] = final_text

                            if final_text:
                                timestamp = int(time.time())
                                audio_path = os.path.join("temp_output", f"speech_{timestamp}.wav")
                                os.makedirs("temp_output", exist_ok=True)
                                translator_engine.generate_audio(final_text, audio_path)
                                st.session_state['audio_path'] = audio_path

                with col_vid_out:
                    st.markdown("**Analyzed Output**")
                    if 'trans_out' in st.session_state:
                        st.video(st.session_state['trans_out'])
                        st.success(f"**Result:** {st.session_state['trans_text']}")
                        if 'audio_path' in st.session_state and os.path.exists(st.session_state['audio_path']):
                            st.audio(st.session_state['audio_path'], format="audio/wav")
                            st.caption("🔊 Audio Player")

# ==================================================
# TAB 4: RESEARCH OR LEARN (Dynamic)
# ==================================================
with tab4:
    if mode_research:
        st.markdown("### 🔬 Research Lab & Reporting")
        st.warning("🔒 Restricted Area: Admin Access Only")

        col_bench_in, col_bench_sets = st.columns([1, 1])
        with col_bench_in:
            bench_file = st.file_uploader("Upload Test Video", type=["mp4"], key="bench_up")
        with col_bench_sets:
            st.info("Benchmark Configuration")
            noise_check = st.checkbox("Inject Gaussian Noise", value=False)

        if bench_file and st.button("📊 Run Analysis & Generate Report", type="primary"):
            bfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            bfile.write(bench_file.read())
            asset_path = os.path.join(os.getcwd(), "assets", "images")

            with st.spinner("Running Tri-Algorithm Analysis..."):
                data = translator_engine.run_research_benchmark(bfile.name, asset_path, noise_check)
                if data:
                    st.divider()
                    st.subheader("1. System Performance Report")
                    avg_m = sum(data['mse_time']) / len(data['mse_time'])
                    avg_n = sum(data['ncc_time']) / len(data['ncc_time'])
                    avg_o = sum(data['orb_time']) / len(data['orb_time'])

                    m1, m2, m3 = st.columns(3)
                    m1.metric("MSE Speed", f"{avg_m:.2f} ms")
                    m2.metric("NCC Speed", f"{avg_n:.2f} ms")
                    m3.metric("ORB Speed", f"{avg_o:.2f} ms")

                    st.subheader("2. Metrics Table")
                    df_bench = pd.DataFrame({
                        "Algorithm": ["MSE", "NCC", "ORB"],
                        "Latency (ms)": [avg_m, avg_n, avg_o],
                        "Noise Tolerance": ["Low", "High", "Excellent"]
                    })
                    st.table(df_bench)

                    report_text = f"""
                    ASL SYSTEM REPORT
                    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    LATENCY SCORES:
                    - MSE (Pixel): {avg_m:.4f} ms/frame
                    - NCC (Template): {avg_n:.4f} ms/frame
                    - ORB (Feature): {avg_o:.4f} ms/frame
                    """
                    st.download_button("📄 Download Official Report (TXT)", report_text, "ASL_Project_Report.txt")

    else:
        st.markdown("### 📚 Learn American Sign Language")
        st.markdown("Enhance your skills with our interactive dictionary and quiz.")

        tab_dict, tab_quiz = st.tabs(["📖 ASL Dictionary", "🧠 Pop Quiz"])

        with tab_dict:
            st.markdown("#### The ASL Alphabet")
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            cols = st.columns(6)
            for i, char in enumerate(alphabet):
                img_path = video_engine.get_image_path(char)
                if img_path:
                    with cols[i % 6]:
                        st.image(img_path, caption=f"Letter: {char.upper()}", use_container_width=True)

        with tab_quiz:
            st.markdown("#### Test Your Knowledge")
            if st.session_state['quiz_current'] is None:
                char = random.choice("abcdefghijklmnopqrstuvwxyz")
                st.session_state['quiz_current'] = char

            target_char = st.session_state['quiz_current']
            img_path = video_engine.get_image_path(target_char)

            col_q_img, col_q_opts = st.columns([1, 1])
            with col_q_img:
                if img_path: st.image(img_path, width=250)

            with col_q_opts:
                st.write("### What letter is this?")
                options = [target_char]
                while len(options) < 4:
                    rand_c = random.choice("abcdefghijklmnopqrstuvwxyz")
                    if rand_c not in options: options.append(rand_c)
                random.shuffle(options)

                with st.form("quiz_form"):
                    choice = st.radio("Select Answer:", [o.upper() for o in options])
                    submit = st.form_submit_button("Submit Answer")

                    if submit:
                        if choice.lower() == target_char:
                            st.balloons()
                            st.success("✅ Correct! +10 Points")
                            st.session_state['quiz_score'] += 10
                            st.session_state['quiz_current'] = None
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Incorrect. It was '{target_char.upper()}'")

            st.metric("Your Score", f"{st.session_state['quiz_score']} pts")

# ==================================================
# TAB 5: FEEDBACK (DYNAMIC ROLE BASED)
# ==================================================
with tab5:
    if mode_research:
        # --- ADMIN VIEW ---
        st.markdown("### 📊 Feedback Dashboard")
        st.markdown("View feedback submitted by users.")

        df_feed = get_feedback_df()

        if not df_feed.empty:
            # 1. Metrics
            avg_rating = df_feed['rating'].mean()
            total_feed = len(df_feed)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⭐ Average Rating</h3>
                    <h1>{avg_rating:.1f}/5</h1>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📝 Total Feedback</h3>
                    <h1>{total_feed}</h1>
                </div>
                """, unsafe_allow_html=True)

            st.write("---")

            # 2. Charts
            rating_counts = df_feed['rating'].value_counts()
            st.bar_chart(rating_counts)

            # 3. Data Table
            st.dataframe(df_feed, use_container_width=True)
        else:
            st.info("No feedback received yet.")

    else:
        # --- USER VIEW ---
        st.markdown("### 📝 Submit Your Feedback")
        st.markdown("We value your thoughts! Help us improve the system.")

        with st.form("feedback_form"):
            rating = st.slider("Rate your experience (1-5):", 1, 5, 5)
            comments = st.text_area("Additional Comments:", placeholder="The translation was accurate...")
            submit_feed = st.form_submit_button("Submit Feedback", type="primary")

            if submit_feed:
                if comments:
                    save_feedback(st.session_state['user_name'], rating, comments)
                    st.success("✅ Thank you! Your feedback has been recorded.")
                else:
                    st.warning("Please write a short comment.")

st.markdown("---")
st.caption("Developed for ASL Research | Powered by OpenCV, MediaPipe, & Streamlit")