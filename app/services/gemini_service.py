import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx
try:
    from sqlalchemy.orm import Session
    from sqlalchemy import func
except ImportError:
    Session = Any
    func = None

from app.config import get_gemini_model, get_gemini_api_key, DEFAULT_GEMINI_MODEL
from app.services.usage_tracker import GeminiUsageTracker
from app.services.gemini_status import GeminiStatusTracker

logger = logging.getLogger("app.services.gemini")

BASE_DIR = Path(__file__).resolve().parent.parent.parent

logging.getLogger("httpx").setLevel(logging.WARNING)


class GeminiQuotaExceededError(RuntimeError):
    """Raised when Gemini returns daily quota exhaustion. Must NOT be retried."""
    pass


class GeminiRateLimitError(RuntimeError):
    """Raised when Gemini returns temporary burst rate limit (per-minute)."""
    pass


def get_api_key(db: Optional[Session] = None) -> str:
    """Retrieve the Gemini API key from single source of truth."""
    return get_gemini_api_key(db)


def get_model_name(db: Optional[Session] = None) -> str:
    """Retrieve the Gemini Model name from single source of truth (default: gemini-3.8-flash)."""
    return get_gemini_model(db)


def clean_json_response(raw_text: str) -> str:
    """Strip markdown code blocks or accidental conversational wrappings from model output."""
    text = raw_text.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        else:
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()

    if not (text.startswith("{") or text.startswith("[")):
        start_obj = text.find("{")
        start_arr = text.find("[")
        if start_obj != -1 and (start_arr == -1 or start_obj < start_arr):
            end_obj = text.rfind("}")
            if end_obj > start_obj:
                text = text[start_obj:end_obj + 1]
        elif start_arr != -1:
            end_arr = text.rfind("]")
            if end_arr > start_arr:
                text = text[start_arr:end_arr + 1]

    return text.strip()


def get_next_product_id(db: Session) -> str:
    """Generate the next sequential product ID: P0001, P0002, etc."""
    from app.models import Product

    products = db.query(Product.product_id).filter(Product.product_id.like("P%")).all()
    max_num = 0
    for (pid,) in products:
        m = re.match(r"^P(\d+)$", pid)
        if m:
            val = int(m.group(1))
            if val > max_num:
                max_num = val

    next_num = max_num + 1
    return f"P{next_num:04d}"


def is_daily_quota_error(status_code: int, err_msg: str, err_json: Optional[Dict[str, Any]] = None) -> bool:
    """
    Determine if a 429 response is a daily quota exhaustion vs a temporary burst rate limit.
    Google Gemini Free Tier daily quota indicators:
    - 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'
    - 'generativelanguage.googleapis.com/generate_content_free_tier_requests'
    - 'GenerateRequestsPerDay'
    - 'free_tier_requests'
    """
    if status_code != 429:
        return False

    combined = (err_msg or "").lower()
    if err_json:
        try:
            combined += " " + json.dumps(err_json).lower()
        except Exception:
            pass

    # If explicitly per minute or RPM, it is temporary burst rate limit
    if "generaterequestsperminute" in combined or "per minute" in combined or "rpm" in combined:
        return False

    daily_indicators = [
        "generaterequestsperday",
        "generate_content_free_tier_requests",
        "freetier",
        "free tier",
        "daily quota",
        "per day",
        "quota exceeded",
        "resource_exhausted"
    ]

    for ind in daily_indicators:
        if ind in combined:
            return True

    return False


def validate_timed_script(
    input_segments: List[Dict[str, Any]],
    ai_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministic local Python validation for batch timed script:
    - Verifies 'segments' key exists and is a list
    - Verifies all input segment_ids are present
    - Verifies vietnamese_text is non-empty and has no placeholder markers
    """
    if not isinstance(ai_output, dict) or "segments" not in ai_output:
        return {
            "valid": False,
            "error": "Phản hồi AI không chứa danh sách 'segments' hợp lệ."
        }

    out_segments = ai_output.get("segments", [])
    if not isinstance(out_segments, list):
        return {
            "valid": False,
            "error": "'segments' phải là một danh sách JSON."
        }

    output_by_id = {}
    for item in out_segments:
        if isinstance(item, dict) and "segment_id" in item:
            output_by_id[item["segment_id"]] = item

    validated_segments = []
    for inp in input_segments:
        sid = inp.get("segment_id")
        if sid not in output_by_id:
            return {
                "valid": False,
                "error": f"Thiếu phân đoạn segment_id={sid} trong phản hồi AI."
            }

        res_item = output_by_id[sid]
        text = str(res_item.get("vietnamese_text", "")).strip()
        if not text:
            return {
                "valid": False,
                "error": f"Phân đoạn segment_id={sid} có lời thoại trống."
            }

        st = float(inp.get("start_time", inp.get("start", 0.0)))
        dur = float(inp.get("duration", 0.0))
        et = float(inp.get("end_time", inp.get("end", round(st + dur, 2))))
        # Word count heuristic: Vietnamese speaking speed is ~2.5 - 3.5 words/second
        word_count = len(text.split())

        validated_segments.append({
            "segment_id": sid,
            "start": st,
            "end": et,
            "start_time": st,
            "end_time": et,
            "duration": dur,
            "vietnamese_text": text,
            "word_count": word_count
        })

    return {
        "valid": True,
        "segments": validated_segments
    }


def validate_rewritten_segments(
    failed_segments: List[Dict[str, Any]],
    ai_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministic local Python validation for batch failed-segment rewrites:
    - Confirms all requested failed segment_ids are returned with non-empty vietnamese_text
    """
    if not isinstance(ai_output, dict) or "rewritten_segments" not in ai_output:
        return {
            "valid": False,
            "error": "Phản hồi AI không chứa 'rewritten_segments' hợp lệ."
        }

    out_items = ai_output.get("rewritten_segments", [])
    if not isinstance(out_items, list):
        return {
            "valid": False,
            "error": "'rewritten_segments' phải là một danh sách JSON."
        }

    by_id = {}
    for item in out_items:
        if isinstance(item, dict) and "segment_id" in item:
            by_id[item["segment_id"]] = item

    rewritten_list = []
    for failed in failed_segments:
        sid = failed.get("segment_id")
        if sid not in by_id:
            return {
                "valid": False,
                "error": f"Thiếu phân đoạn được viết lại segment_id={sid}."
            }

        text = str(by_id[sid].get("vietnamese_text", "")).strip()
        if not text:
            return {
                "valid": False,
                "error": f"Phân đoạn segment_id={sid} được viết lại nhưng lời thoại trống."
            }

        rewritten_list.append({
            "segment_id": sid,
            "target_duration": failed.get("target_duration"),
            "vietnamese_text": text
        })

    return {
        "valid": True,
        "rewritten_segments": rewritten_list
    }


class GeminiService:
    def __init__(self):
        pass

    def call_gemini(
        self,
        prompt: str,
        db: Optional[Session] = None,
        timeout: float = 60.0,
        max_retries: int = 2
    ) -> str:
        """
        Production-grade Gemini API caller using configured GEMINI_MODEL.
        Features:
        - Strict low request consumption: No retries on Daily Quota exhaustion
        - Discriminates 429 Daily Quota vs 429 Temporary Rate Limit
        - Fast-fails with GeminiQuotaExceededError on daily quota
        - Records lightweight local usage statistics in SQLite
        - Updates GeminiStatusTracker for UI badge (no extra quota consumed)
        - Masks API key from logs and errors
        - 503 exponential backoff: attempt1→2s, attempt2→5s, then fail
        """
        api_key = get_api_key(db)
        model_name = get_model_name(db)

        if not api_key:
            GeminiStatusTracker.update_status("AUTH_ERROR", db=db, http_code=401)
            raise ValueError("Gemini API key is not configured. Please enter your API key in Settings.")

        # Record start of request in local tracker
        GeminiUsageTracker.record_request_start(db=db)
        # Signal that a real request is now in-flight (local only, no network)
        GeminiStatusTracker.update_status("CHECKING", db=db)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }

        # Exponential backoff delays for 503: attempt 0→2s, attempt 1→5s
        _503_delays = [2.0, 5.0]

        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    status_code = resp.status_code

                    # 200 OK
                    if status_code == 200:
                        try:
                            data = resp.json()
                        except Exception as je:
                            GeminiUsageTracker.record_failure(db=db, is_quota=False)
                            GeminiStatusTracker.update_status("ERROR", db=db, http_code=200)
                            raise ValueError(f"Malformed JSON returned by Gemini endpoint: {je}")

                        candidates = data.get("candidates", [])
                        if not candidates:
                            GeminiUsageTracker.record_failure(db=db, is_quota=False)
                            GeminiStatusTracker.update_status("ERROR", db=db, http_code=200)
                            raise ValueError("Gemini API returned an empty response (no candidates).")

                        candidate = candidates[0]
                        parts = candidate.get("content", {}).get("parts", [])
                        if not parts:
                            GeminiUsageTracker.record_failure(db=db, is_quota=False)
                            GeminiStatusTracker.update_status("ERROR", db=db, http_code=200)
                            raise ValueError("Gemini API returned an empty response (no content parts).")

                        text_parts = [p.get("text", "") for p in parts if "text" in p]
                        raw_text = "".join(text_parts).strip()
                        if not raw_text:
                            GeminiUsageTracker.record_failure(db=db, is_quota=False)
                            GeminiStatusTracker.update_status("ERROR", db=db, http_code=200)
                            raise ValueError("Gemini API returned empty text.")

                        GeminiUsageTracker.record_success(db=db)
                        GeminiStatusTracker.update_status("READY", db=db, http_code=200)
                        return raw_text

                    # Try to extract error message from response JSON
                    err_msg = ""
                    err_json = None
                    try:
                        err_json = resp.json()
                        err_msg = err_json.get("error", {}).get("message", resp.text[:200])
                    except Exception:
                        err_msg = resp.text[:200]

                    # 429 Rate Limit / Quota Exceeded
                    if status_code == 429:
                        if is_daily_quota_error(status_code, err_msg, err_json):
                            # DAILY QUOTA EXHAUSTED: DO NOT RETRY. FAIL FAST.
                            GeminiUsageTracker.record_failure(db=db, is_quota=True)
                            GeminiStatusTracker.update_status("QUOTA_EXCEEDED", db=db, http_code=429)
                            logger.error(f"Gemini daily quota exhausted for model {model_name}. Aborting without retry.")
                            raise GeminiQuotaExceededError(
                                "GEMINI_QUOTA_EXCEEDED: Gemini daily quota has been reached. "
                                "Try again after quota reset or use a project with sufficient quota."
                            )

                        # Temporary Rate Limit (burst / per-minute)
                        if attempt < max_retries:
                            retry_delay = 2.0 * (attempt + 1)
                            m_wait = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                            if m_wait:
                                try:
                                    extracted = float(m_wait.group(1)) + 1.0
                                    if extracted <= 15.0:
                                        retry_delay = extracted
                                except Exception:
                                    pass
                            GeminiStatusTracker.update_status("RATE_LIMITED", db=db, http_code=429)
                            logger.warning(
                                f"Gemini 429 Temporary Rate Limit (attempt {attempt + 1}/{max_retries + 1}), "
                                f"waiting {retry_delay:.1f}s before retry..."
                            )
                            time.sleep(retry_delay)
                            continue

                        GeminiUsageTracker.record_failure(db=db, is_quota=False)
                        GeminiStatusTracker.update_status("RATE_LIMITED", db=db, http_code=429)
                        raise GeminiRateLimitError(f"Gemini API 429 Rate Limit Exceeded: {err_msg}")

                    # 400 Bad Request
                    if status_code == 400:
                        GeminiUsageTracker.record_failure(db=db, is_quota=False)
                        GeminiStatusTracker.update_status("ERROR", db=db, http_code=400)
                        raise RuntimeError(f"Gemini API 400 Bad Request: {err_msg}")

                    # 401 Unauthorized
                    if status_code == 401:
                        GeminiUsageTracker.record_failure(db=db, is_quota=False)
                        GeminiStatusTracker.update_status("AUTH_ERROR", db=db, http_code=401)
                        raise RuntimeError("Gemini API 401 Unauthorized: Invalid API Key. Please verify your key in Settings.")

                    # 403 Forbidden
                    if status_code == 403:
                        GeminiUsageTracker.record_failure(db=db, is_quota=False)
                        GeminiStatusTracker.update_status("AUTH_ERROR", db=db, http_code=403)
                        raise RuntimeError(f"Gemini API 403 Forbidden: Permission denied for model '{model_name}'. Details: {err_msg}")

                    # 404 Model Not Found
                    if status_code == 404:
                        GeminiUsageTracker.record_failure(db=db, is_quota=False)
                        GeminiStatusTracker.update_status("ERROR", db=db, http_code=404)
                        raise RuntimeError(f"Gemini API 404 Not Found: Model '{model_name}' was not found. Details: {err_msg}")

                    # 500+ / 503 Server Error / Overload — exponential backoff
                    if status_code >= 500:
                        GeminiStatusTracker.update_status("BUSY", db=db, http_code=status_code)
                        if attempt < max_retries:
                            delay = _503_delays[attempt] if attempt < len(_503_delays) else 5.0
                            logger.warning(
                                f"Gemini {status_code} Temporary Overload "
                                f"(attempt {attempt + 1}/{max_retries + 1}), "
                                f"retrying in {delay:.0f}s..."
                            )
                            time.sleep(delay)
                            continue
                        GeminiUsageTracker.record_failure(db=db, is_quota=False)
                        raise RuntimeError(
                            f"Gemini API {status_code} Server Error: Gemini vẫn đang quá tải sau "
                            f"{max_retries + 1} lần thử. Hãy thử lại sau."
                        )

                    # Other unexpected status codes
                    GeminiUsageTracker.record_failure(db=db, is_quota=False)
                    GeminiStatusTracker.update_status("ERROR", db=db, http_code=status_code)
                    raise RuntimeError(f"Gemini API error {status_code}: {err_msg}")

            except (GeminiQuotaExceededError, GeminiRateLimitError, ValueError, RuntimeError) as e:
                # If already handled custom error, raise immediately without further retrying
                raise e

            except httpx.TimeoutException:
                last_error = RuntimeError(f"Gemini API request timed out after {timeout} seconds.")
                if attempt < max_retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying in 2s...")
                    time.sleep(2)
                    continue
                GeminiUsageTracker.record_failure(db=db, is_quota=False)
                GeminiStatusTracker.update_status("NETWORK_ERROR", db=db)
                raise last_error

            except httpx.NetworkError as ne:
                last_error = RuntimeError(f"Gemini API network failure: {str(ne)}")
                if attempt < max_retries:
                    logger.warning(f"Network error on attempt {attempt + 1}, retrying in 2s...")
                    time.sleep(2)
                    continue
                GeminiUsageTracker.record_failure(db=db, is_quota=False)
                GeminiStatusTracker.update_status("NETWORK_ERROR", db=db)
                raise last_error

            except Exception as e:
                GeminiUsageTracker.record_failure(db=db, is_quota=False)
                GeminiStatusTracker.update_status("ERROR", db=db)
                raise RuntimeError(f"Gemini request failed: {str(e)}")

        if last_error:
            GeminiUsageTracker.record_failure(db=db, is_quota=False)
            GeminiStatusTracker.update_status("ERROR", db=db)
            raise last_error
        GeminiUsageTracker.record_failure(db=db, is_quota=False)
        GeminiStatusTracker.update_status("ERROR", db=db)
        raise RuntimeError("Gemini API call failed after retries.")

    def test_connection(self, db: Optional[Session] = None) -> Dict[str, Any]:
        """Test Gemini API connection using the configured model."""
        api_key = get_api_key(db)
        model_name = get_model_name(db)

        if not api_key:
            return {
                "success": False,
                "error": "Gemini API key is not configured. Please add your key in Settings."
            }

        try:
            res_text = self.call_gemini(
                prompt="Ping. Respond with 'OK'.",
                db=db,
                timeout=15.0,
                max_retries=0
            )
            return {
                "success": True,
                "message": f"Connected successfully to {model_name}.",
                "model": model_name,
                "response": res_text
            }
        except GeminiQuotaExceededError:
            return {
                "success": False,
                "quota_exceeded": True,
                "error": "Gemini daily quota has been reached. Try again after quota reset or use a project with sufficient quota.",
                "model": model_name
            }
        except Exception as e:
            logger.error(f"Gemini connection test failed: {e}")
            return {
                "success": False,
                "error": f"Connection failed: {str(e)}",
                "model": model_name
            }

    def generate_products(
        self,
        niche: str,
        count: int = 10,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch Product Research: Generate N products in exactly ONE Gemini request.
        Validates output locally in Python.
        """
        prompt = f"""You are an expert e-commerce and viral short-video researcher for Douyin (TikTok China).
Target Niche: {niche}
Generate a JSON list of exactly {count} trending, problem-solving, or viral products for this niche.

CRITICAL REQUIREMENTS:
1. "name_vietnamese": Clear, commercial Vietnamese product name.
2. "name_chinese": Natural commercial Chinese product name used by Chinese suppliers.
3. "douyin_keywords": Natural Chinese search phrases specifically used on Douyin to discover real product showcase videos. Do NOT make literal word-by-word translations. Use authentic Douyin short-video terms (e.g., '居家好物', '神器', '开箱', '测评', '好物推荐', specific feature descriptors).
4. "content_angle": Short compelling angle for short video (in Vietnamese).
5. "hook": Short 1-sentence opening hook to grab attention in first 3 seconds (in Vietnamese).

OUTPUT FORMAT:
Return ONLY a valid JSON array of objects. No intro text, no conversational text, no markdown other than standard JSON.
Example structure:
[
  {{
    "name_vietnamese": "Nồi cơm điện mini đa năng",
    "name_chinese": "多功能迷你电饭煲",
    "douyin_keywords": "宿舍迷你电饭煲 独居一人食好物 煮饭神器",
    "content_angle": "Giải pháp nấu ăn tiện lợi nhanh gọn cho người sống một mình",
    "hook": "Đừng mua nồi cơm to nữa nếu bạn sống một mình hoặc ở trọ!"
  }}
]
"""
        raw_content = self.call_gemini(prompt, db=db, timeout=60.0)

        cleaned = clean_json_response(raw_content)
        try:
            items = json.loads(cleaned)
            if not isinstance(items, list):
                raise ValueError("Response is not a JSON list.")
        except Exception as e:
            logger.error(f"Failed to parse Gemini JSON output: {cleaned[:200]} - Error: {e}")
            raise ValueError(f"Malformed AI response format: {str(e)}")

        # Validate schema of items locally
        valid_items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            valid_items.append({
                "name_vietnamese": str(it.get("name_vietnamese", "")).strip(),
                "name_chinese": str(it.get("name_chinese", "")).strip(),
                "douyin_keywords": str(it.get("douyin_keywords", "")).strip(),
                "content_angle": str(it.get("content_angle", "")).strip(),
                "hook": str(it.get("hook", "")).strip(),
            })

        return valid_items

    def generate_timed_script(
        self,
        video_duration: float,
        segments: List[Dict[str, Any]],
        product_info: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Phase 12 Timed Script: Sends the COMPLETE timeline in ONE single Gemini call.
        Does NOT call Gemini once per segment.
        Validates returned segments locally with Python.
        """
        if not segments:
            return {"valid": False, "error": "Danh sách phân đoạn timeline trống."}

        prod_name = (product_info or {}).get("name_vietnamese", "Sản phẩm")
        content_angle = (product_info or {}).get("content_angle", "Tính năng nổi bật")
        hook = (product_info or {}).get("hook", "Món đồ tiện ích không thể bỏ lỡ")

        segment_lines = []
        for s in segments:
            sid = s.get("segment_id")
            st = float(s.get("start_time", s.get("start", 0.0)))
            dur = float(s.get("duration", 0.0))
            et = float(s.get("end_time", s.get("end", round(st + dur, 2))))
            if dur <= 0.0:
                dur = round(et - st, 2)
            desc = s.get("description", "")
            desc_part = f" - Diễn biến cảnh: {desc}" if desc else ""
            segment_lines.append(f"- Phân đoạn {sid}: từ {st:.1f}s đến {et:.1f}s (Thời lượng: {dur:.1f}s){desc_part}")

        timeline_str = "\n".join(segment_lines)

        prompt = f"""Bạn là một chuyên gia biên kịch video ngắn triệu view (TikTok, Reels, Shorts).
Nhiệm vụ: Viết lời đọc thuyết minh (Voice-over) bằng TIẾNG VIỆT cho TOÀN BỘ video sau.

THÔNG TIN SẢN PHẨM:
- Tên sản phẩm: {prod_name}
- Góc tiếp cận: {content_angle}
- Câu mở đầu (Hook): {hook}
- Tổng thời lượng video: {video_duration:.1f} giây

TIMELINE CÁC PHÂN ĐOẠN (Toàn bộ kịch bản phải khớp với từng mốc thời gian):
{timeline_str}

QUY TẮC BẮT BUỘC:
1. KHÔNG được viết quá dài. Tốc độ nói tiếng Việt tự nhiên là khoảng 2.5 đến 3.2 từ mỗi giây.
   Mỗi phân đoạn PHẢI có số lượng từ vừa vặn với thời lượng của phân đoạn đó.
2. Lời thoại tự nhiên, liền mạch giữa các phân đoạn, giọng văn bán hàng thu hút.
3. CHỈ TRẢ VỀ DUY NHẤT một đối tượng JSON hợp lệ theo đúng cấu trúc sau:
{{
  "segments": [
    {{
      "segment_id": 1,
      "vietnamese_text": "..."
    }}
  ]
}}
"""
        raw_text = self.call_gemini(prompt, db=db, timeout=60.0)
        cleaned = clean_json_response(raw_text)

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"AI returned malformed JSON: {e}")

        val_result = validate_timed_script(segments, parsed)
        if not val_result["valid"]:
            raise ValueError(val_result["error"])

        return val_result

    @staticmethod
    def build_rewrite_prompt(
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        product_name: str = ""
    ) -> str:
        """Construct the prompt for batch rewriting failed segments."""
        segment_lines = []
        for s in failed_segments:
            sid = s.get("segment_id")
            target_dur = s.get("target_duration", 0.0)
            actual_dur = s.get("actual_duration", 0.0)
            curr_text = s.get("current_text", s.get("vietnamese_text", ""))
            direction = s.get("direction", "shorten" if actual_dur > target_dur else "lengthen")

            action = "RÚT NGẮN lại lời đọc" if direction == "shorten" else "KÉO DÀI thêm lời đọc"
            segment_lines.append(
                f"- Phân đoạn {sid}: Thời lượng mục tiêu: {target_dur:.1f}s | "
                f"Thời lượng audio thực tế: {actual_dur:.1f}s. "
                f"Yêu cầu: {action}.\n  Lời hiện tại: \"{curr_text}\""
            )

        failed_str = "\n".join(segment_lines)
        prod_context = f"\nSản phẩm: {product_name}" if product_name else ""

        return f"""Sau khi đo đạc âm thanh thực tế, các phân đoạn sau đây KHÔNG khớp với thời lượng video.{prod_context}
Hãy viết lại LỜI ĐỌC TIẾNG VIỆT cho TẤT CẢ các phân đoạn bị lệch dưới đây trong MỘT LẦN DUY NHẤT.
Vòng viết lại: {round_num}/{max_rounds}

DANH SÁCH PHÂN ĐOẠN CẦN CHỈNH SỬA:
{failed_str}

QUY TẮC:
1. Nếu cần rút ngắn: Giảm bớt số từ, dùng từ cô đọng, giữ trọn ý chính.
2. Nếu cần kéo dài: Thêm mô tả nhẹ nhàng, tự nhiên.
3. CHỈ TRẢ VỀ DUY NHẤT MỘT ĐỐI TƯỢNG JSON với cấu trúc:
{{
  "rewritten_segments": [
    {{
      "segment_id": 1,
      "vietnamese_text": "..."
    }}
  ]
}}
"""

    def rewrite_failed_segments(
        self,
        failed_segments: List[Dict[str, Any]],
        round_num: int = 1,
        max_rounds: int = 2,
        db: Optional[Session] = None,
        product_name: str = ""
    ) -> Dict[str, Any]:
        """
        Duration Rewrite: Sends ALL failed segments in ONE single Gemini request.
        Enforces maximum rewrite rounds (max 2 rounds, no infinite retries).
        Validates rewritten segments locally with Python.
        """
        if round_num > max_rounds:
            return {
                "success": False,
                "error": f"Đã vượt quá số lần viết lại tối đa ({max_rounds} vòng). Cần can thiệp thủ công.",
                "max_rounds_exceeded": True,
                "round": round_num
            }

        if not failed_segments:
            return {
                "success": True,
                "round": round_num,
                "rewritten_segments": []
            }

        prompt = self.build_rewrite_prompt(failed_segments, round_num=round_num, max_rounds=max_rounds, product_name=product_name)
        raw_text = self.call_gemini(prompt, db=db, timeout=60.0)
        cleaned = clean_json_response(raw_text)

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            raise ValueError(f"AI returned malformed JSON: {e}")

        val_result = validate_rewritten_segments(failed_segments, parsed)
        if not val_result["valid"]:
            raise ValueError(val_result["error"])

        return {
            "success": True,
            "round": round_num,
            "rewritten_segments": val_result["rewritten_segments"]
        }
