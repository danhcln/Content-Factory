# 🏭 AI CONTENT FACTORY - LOCAL SYSTEM

> Hệ thống quản lý & tự động hóa sản xuất video đa nền tảng (7 nền tảng) chạy cục bộ trên Windows.

---

## 📋 MỤC LỤC
1. [Yêu cầu phần mềm (Required Software)](#1-yêu-cầu-phần-mềm)
2. [Cài đặt Python (Python Setup)](#2-cài-đặt-python)
3. [Cài đặt thư viện phụ thuộc (Dependencies)](#3-cài-đặt-thư-viện)
4. [Cấu hình Google Gemini API Key](#4-cấu-hình-gemini-api)
5. [Cài đặt & cấu hình VieNeu-TTS](#5-cài-đặt-vieneu-tts)
6. [Khởi động với START.bat](#6-khởi-động-với-startbat)
7. [Hướng dẫn Nghiên Cứu Sản Phẩm (Research)](#7-nghiên-cứu-sản-phẩm)
8. [Nhập & Phê Duyệt Video (Video Import & Review)](#8-nhập--phê-duyệt-video)
9. [Quy trình dựng CapCut (CapCut Workflow)](#9-quy-trình-dựng-capcut)
10. [Nơi xuất video Final (Exporting Final Videos)](#10-nơi-xuất-video-final)
11. [Tự động tạo nội dung 7 nền tảng (Caption Generation)](#11-tự-động-tạo-nội-dung-7-nền-tảng)
12. [Hàng đợi đăng bài (Publishing Queue - 7/7)](#12-hàng-đợi-đăng-bài)
13. [Các lỗi thường gặp và cách khắc phục](#13-các-lỗi-thường-gặp)
14. [Development / GitHub Setup](#14-development--github-setup)

---

### 1. Yêu cầu phần mềm
- **Hệ điều hành**: Windows 10 hoặc Windows 11 (64-bit).
- **Python**: Phiên bản Python 3.10 trở lên (khuyên dùng Python 3.10 hoặc 3.11/3.14).
- **Trình duyệt**: Google Chrome, Microsoft Edge, hoặc Brave.
- **CapCut PC**: Phiên bản CapCut PC để dựng video thủ công.

---

### 2. Cài đặt Python
1. Tải Python từ trang chủ [python.org](https://www.python.org/downloads/).
2. Khi chạy bộ cài installer, **BẮT BUỘC ĐÁNH DẤU CHỌN**:
   - `Add Python to PATH` (hoặc `Add python.exe to PATH`).
3. Mở Command Prompt (`cmd`) hoặc PowerShell, gõ:
   ```cmd
   python --version
   ```
   Nếu hiển thị phiên bản Python (ví dụ: `Python 3.11.x`), bạn đã cài đặt thành công.

---

### 3. Cài đặt thư viện
Mở Command Prompt trong thư mục dự án và chạy lệnh:
```cmd
pip install -r requirements.txt
```

---

### 4. Cấu hình Gemini API
1. Truy cập [Google AI Studio](https://aistudio.google.com/) để tạo API Key miễn phí.
2. Bạn có thể cấu hình bằng 2 cách:
   - **Cách 1 (Khuyên dùng)**: Mở giao diện ứng dụng tại `http://localhost:8000/settings`, dán API Key vào ô **Gemini API Key** rồi nhấn **Lưu Cài Đặt**.
   - **Cách 2**: Mở file `.env` bằng Notepad và điền:
     ```env
     GEMINI_API_KEY=AIzaSy...
     GEMINI_MODEL=gemini-3.8-flash
     ```

---

### 5. Cài đặt VieNeu-TTS
Hệ thống sử dụng **VieNeu-TTS** chạy hoàn toàn cục bộ trên máy tính (không tốn chi phí và không phụ thuộc dịch vụ ngoài như ElevenLabs):
- Repository: [https://github.com/pnnbao97/VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS)
- Mặc định hệ thống sử dụng model ONNX/Torch chạy nội bộ để tạo giọng đọc tự nhiên chuẩn tiếng Việt (Bắc / Nam).

---

### 6. Khởi động với START.bat
Chỉ cần **nháy đúp chuột vào file `START.bat`**:
1. File `.bat` sẽ tự động kiểm tra môi trường Python.
2. Tự động khởi tạo cấu trúc thư mục dữ liệu và database SQLite `data/content.db`.
3. Bật máy chủ FastAPI tại `http://127.0.0.1:8000`.
4. Tự động mở trình duyệt web đến trang quản trị **Dashboard**.

---

### 7. Nghiên Cứu Sản Phẩm
1. Vào menu **Research** bên thanh bên trái.
2. Nhập ngành hàng (Niche), ví dụ: `Đồ gia dụng thông minh`.
3. Chọn số lượng sản phẩm (5, 10, 20...).
4. Bấm **RUN RESEARCH**.
5. Gemini sẽ tự động tạo tên tiếng Việt, tên tiếng Trung thương mại, từ khóa Douyin tự nhiên, Hook và Content Angle, sau đó lưu thành các mã `P0001`, `P0002`...

---

### 8. Nhập & Phê Duyệt Video
1. Từ sản phẩm đã có, hệ thống hỗ trợ quét hoặc nhập URL Douyin thủ công.
2. Vào **Review Queue** để kiểm tra danh sách video tìm thấy (`FOUND`).
3. Bạn có thể xem trước URL, lượt xem (nếu có), bấm **Approve** từng video hoặc chọn nhiều video và nhấn **APPROVE SELECTED**.
4. Video được duyệt sẽ chuyển sang trạng thái `APPROVED` để chuẩn bị tải về.

---

### 9. Quy trình dựng CapCut
1. Sau khi video gốc được tải về và VieNeu-TTS tạo xong voice-over, hệ thống tự động đóng gói tại:
   ```
   downloads/packages/V0001/
     ├── original.mp4    (Video gốc tham khảo)
     ├── voice.wav       (Giọng đọc tiếng Việt)
     ├── script.txt      (Kịch bản lời đọc)
     └── info.json       (Thông tin chi tiết)
   ```
2. Trạng thái chuyển thành `WAITING_CAPCUT`.
3. Người dùng mở CapCut PC, kéo các file trong thư mục gói vào để dựng video theo ý thích.

---

### 10. Nơi xuất video Final
Khi dựng xong trong CapCut PC, bạn bấm **Export** và lưu video vào thư mục:
```
downloads/final/
```
Đặt tên file theo đúng mã Video ID:
- `V0001.mp4`
- `V0002.mp4`
- ...

---

### 11. Tự động tạo nội dung 7 nền tảng
Folder Watcher (chạy nền) tự động phát hiện khi file `V0001.mp4` xuất hiện:
1. Đọc kịch bản gốc tương ứng của `V0001`.
2. Gửi kịch bản đến Gemini với **duy nhất 1 request**.
3. Nhận về định dạng chuẩn JSON cho cả 7 nền tảng:
   - Facebook Cá Nhân (giọng điệu chia sẻ tự nhiên)
   - Facebook Fanpage (chuyên nghiệp, nổi bật tính năng)
   - TikTok (ngắn gọn, mở đầu lôi cuốn)
   - Threads (tự nhiên, ngắn gọn)
   - Instagram (hashtag chọn lọc)
   - Shopee Video (tập trung sản phẩm)
   - YouTube Shorts (tiêu đề + mô tả chuẩn SEO ngắn)
4. Lưu vào bảng `CONTENT` và chuyển trạng thái sang `CONTENT_READY`.

---

### 12. Hàng đợi đăng bài
1. Vào mục **Publishing Queue**.
2. Với mỗi video đã sẵn sàng:
   - Bấm nút **Copy Caption** hoặc **Copy Title / Description**.
   - Bấm nút **Open Final Video** để xem nhanh video.
3. Sau khi đăng lên nền tảng nào, bấm tick chọn nền tảng đó.
4. Tiến độ tự động nhảy: `1/7`, `2/7` ... Khi đạt `7/7`, trạng thái chuyển thành `COMPLETED`.

---

### 13. Các lỗi thường gặp
- **Lỗi `ModuleNotFoundError`**: Mở cmd chạy `pip install -r requirements.txt`.
- **Lỗi kết nối Gemini**: Vào menu **Settings**, kiểm tra xem đã dán đúng API Key chưa và nhấn **TEST GEMINI**.
- **Video xuất CapCut không tự nhận**: Kiểm tra tên file trong `downloads/final/` xem đã đúng định dạng `V0001.mp4` chưa (chữ V in hoa kèm 4 số).
- **Xem nhật ký chi tiết**: Mở file `logs/app.log` để xem toàn bộ thông tin chi tiết lỗi nếu có.

---

### 14. Development / GitHub Setup

Hướng dẫn thiết lập môi trường phát triển cho lập trình viên:

1. **Clone repository**:
   ```cmd
   git clone <REPO_URL>
   cd Workflow_video
   ```
2. **Create Python environment**:
   Tạo và kích hoạt môi trường ảo Python 3.11:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. **Install requirements**:
   Cài đặt các gói thư viện dự án:
   ```cmd
   pip install -r requirements.txt
   ```
4. **Create .env from .env.example**:
   Sao chép template cấu hình:
   ```cmd
   copy .env.example .env
   ```
   Mở file `.env` và điền `GEMINI_API_KEY` cùng các thông số môi trường của bạn.
5. **Configure local FFmpeg**:
   Cài đặt FFmpeg trên máy và khai báo đường dẫn `FFMPEG_PATH` và `FFPROBE_PATH` trong `.env`.
6. **Configure VieNeu local environment**:
   Thiết lập môi trường ảo nội bộ độc lập `.venv_vieneu_new` dành riêng cho VieNeu-TTS.
7. **Run application**:
   Khởi động ứng dụng bằng file batch:
   ```cmd
   START.bat
   ```
8. **Database is local and not committed**:
   Database SQLite `data/content.db` lưu trữ dữ liệu runtime cục bộ, không commit lên Git. Mỗi lập trình viên có database riêng biệt.
9. **downloads/ is local and not committed**:
   Thư mục `downloads/` (bao gồm `original/`, `packages/`, `final/`), cùng các thư mục `logs/`, `temp/`, và `test_outputs/` là dữ liệu sinh ra trong quá trình chạy, không đưa lên GitHub.
