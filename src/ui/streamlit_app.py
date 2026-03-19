# src/ui/app.py
import streamlit as st
import requests
import json
import uuid
import time

# --- CẤU HÌNH ---
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Vietnamese RAG Dynamic", page_icon="📑", layout="wide")

st.markdown("""
<style>
    .source-box { background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-size: 0.9em; margin-top: 15px; border-left: 3px solid #4CAF50; }
    .error-text { color: #d32f2f; font-weight: bold; }
    .status-box { padding: 10px; border-radius: 5px; background-color: #e3f2fd; color: #1565c0; font-weight: bold; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- SESSION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8] # Mã định danh duy nhất cho user này
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False

# --- SIDEBAR UI (UPLOAD & MANAGEMENT) ---
with st.sidebar:
    st.title("🗂️ Quản lý Tài liệu")
    st.caption(f"Phiên làm việc: `{st.session_state.session_id}`")
    
    uploaded_file = st.file_uploader("Tải lên PDF để AI học", type="pdf")
    
    # Nút dọn dẹp
    if st.button("🗑️ Xóa dữ liệu & Tạo phiên mới", use_container_width=True):
        if st.session_state.is_ready:
            requests.delete(f"{API_BASE_URL}/session/{st.session_state.session_id}")
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.session_state.is_ready = False
        st.rerun()

    # file upload handling
    if uploaded_file is not None and not st.session_state.is_ready:
        st.markdown("---")
        status_placeholder = st.empty()
        
        # send file to backend
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        data = {"session_id": st.session_state.session_id}
        
        try:
            res = requests.post(f"{API_BASE_URL}/upload", files=files, data=data)
            if res.status_code == 200:
                # polling backend for status updates
                with st.spinner("AI đang đọc tài liệu..."):
                    while True:
                        status_res = requests.get(f"{API_BASE_URL}/status/{st.session_state.session_id}").json()
                        current_status = status_res.get("status", "")
                        
                        status_placeholder.markdown(f"<div class='status-box'>⏳ {current_status}</div>", unsafe_allow_html=True)
                        
                        if current_status == "Hoàn tất":
                            st.session_state.is_ready = True
                            status_placeholder.success("✅ Hệ thống đã học xong tài liệu! Bạn có thể bắt đầu hỏi.")
                            time.sleep(1)
                            st.rerun() # lock chat ui
                            break
                        elif "Lỗi" in current_status:
                            status_placeholder.error(current_status)
                            break
                        
                        time.sleep(1.5) # timeout polling
            else:
                st.error("Lỗi khi gửi file lên server.")
        except requests.exceptions.ConnectionError:
            st.error("Không thể kết nối tới Backend. Hãy chắc chắn FastAPI đang chạy.")


# --- MAIN UI ---
st.title("🤖 Trợ lý AI Phân tích Văn bản")

if not st.session_state.is_ready:
    st.info("👈 Vui lòng tải lên một tài liệu PDF ở cột bên trái để bắt đầu trò chuyện.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Đặt câu hỏi về tài liệu bạn vừa tải lên..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            full_response = ""
            sources_html = ""
            has_error = False

            try:
                response_placeholder.markdown("*(Đang tìm kiếm...)*")
                
                # Gắn thêm session_id vào query API
                ask_url = f"{API_BASE_URL}/ask?query={prompt}&session_id={st.session_state.session_id}"
                with requests.get(ask_url, stream=True, timeout=120) as r:
                    if r.status_code != 200:
                        has_error = True
                        full_response = f"<span class='error-text'>Lỗi kết nối Backend (Status {r.status_code}).</span>"
                        response_placeholder.markdown(full_response, unsafe_allow_html=True)
                    else:
                        for line in r.iter_lines():
                            if line:
                                try:
                                    chunk = json.loads(line.decode("utf-8"))
                                    chunk_type = chunk.get("type")
                                    
                                    if chunk_type == "error":
                                        has_error = True
                                        full_response = f"<span class='error-text'>⚠️ {chunk.get('message')}</span>"
                                        response_placeholder.markdown(full_response, unsafe_allow_html=True)
                                        break
                                    elif chunk_type == "sources":
                                        sources = chunk.get("data", [])
                                        if sources:
                                            pages = sorted(list(set([str(s.get('page', '?')) for s in sources])))
                                            sources_html = f"<div class='source-box'><b>📚 Nguồn:</b> Tìm thấy tại Trang: {', '.join(pages)}</div>"
                                            response_placeholder.empty()
                                    elif chunk_type == "content":
                                        if full_response == "": response_placeholder.empty()
                                        full_response += chunk.get("data", "")
                                        response_placeholder.markdown(full_response + "▌")
                                except json.JSONDecodeError:
                                    continue

                if not has_error:
                    final_output = full_response
                    response_placeholder.markdown(final_output)
                    if sources_html:
                        sources_placeholder.markdown(sources_html, unsafe_allow_html=True)
                        final_output += f"\n\n{sources_html}"
                    st.session_state.messages.append({"role": "assistant", "content": final_output})
                else:
                     st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Lỗi hệ thống."})

            except requests.exceptions.ConnectionError:
                st.error("Lỗi kết nối API.")