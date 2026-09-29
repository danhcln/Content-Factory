import logging
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Product
from app.services.ai import (
    get_ai_manager,
    AIProviderError,
    AIQuotaExceededError,
    AIRateLimitError,
    AIAuthenticationError,
    AIPermissionError,
    AIModelNotFoundError,
    AIServiceUnavailableError,
    AITimeoutError,
    AINetworkError,
)
from app.services.ai.base import AIInvalidResponseError
from app.services.gemini_service import (
    GeminiService,
    get_next_product_id,
    GeminiQuotaExceededError,
    sanitize_error_message,
)
from app.services.gemini_status import GeminiStatusTracker

logger = logging.getLogger("app.routes.research")

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

router = APIRouter(tags=["research"])


def _get_research_provider_info(db: Session) -> dict:
    """Load current active provider info for Research UI — local only, zero network calls."""
    from app.services.ai import get_ai_manager
    from app.services.ai.manager import get_provider_key_configured
    from app.config import get_active_ai_provider, get_active_ai_model

    manager = get_ai_manager()
    active_prov = manager.get_active_provider(db=db)
    pid = active_prov.provider_id
    pname = "Gemini" if pid == "gemini" else active_prov.display_name
    model = get_active_ai_model(db=db)
    configured = get_provider_key_configured(pid, db=db)

    gemini_status = GeminiStatusTracker.get_status(db=db)
    if pid == "gemini":
        status_code = gemini_status.get("status", "READY" if configured else "AUTH_ERROR")
        if not configured:
            status_code = "AUTH_ERROR"
            status_msg = "Chưa cấu hình Gemini API Key"
            status_hint = "Vui lòng nhập API Key trong Cài đặt."
        else:
            status_msg = gemini_status.get("message", "Gemini sẵn sàng")
            status_hint = gemini_status.get("hint", f"Mô hình: {model}. Mỗi lượt chạy gửi đúng 1 yêu cầu duy nhất tới Gemini.")
    else:
        status_code = "READY" if configured else "AUTH_ERROR"
        status_msg = f"{pname} đã cấu hình và sẵn sàng" if configured else f"Chưa cấu hình API Key cho {pname}"
        status_hint = (
            f"Mô hình: {model}. Mỗi lượt chạy gửi đúng 1 yêu cầu duy nhất tới {pname}."
            if configured
            else f"Vui lòng thiết lập API Key cho {pname} trong Cài đặt."
        )

    return {
        "provider_id": pid,
        "provider_name": pname,
        "model": model,
        "configured": configured,
        "status": status_code,
        "status_msg": status_msg,
        "status_hint": status_hint,
        "gemini_status": gemini_status
    }


def _get_gemini_status(db: Session) -> dict:
    """Load current local Gemini status for display — no network calls."""
    return GeminiStatusTracker.get_status(db=db)


def normalize_niche(niche: str) -> str:
    """Deterministic local normalization for niche strings."""
    if not niche:
        return ""
    return re.sub(r"\s+", " ", niche.strip().lower())


def get_cached_products_for_niche(db: Session, niche: str) -> List[Product]:
    """Retrieve existing products matching normalized niche without calling AI."""
    norm = normalize_niche(niche)
    if not norm:
        return []
    all_prods = db.query(Product).order_by(Product.id.asc()).all()
    return [p for p in all_prods if normalize_niche(p.niche) == norm]


@router.get("/research", response_class=HTMLResponse)
def get_research_page(
    request: Request,
    niche: Optional[str] = None,
    product_count: int = 10,
    fresh: bool = False,
    db: Session = Depends(get_db)
):
    cached_count = len(get_cached_products_for_niche(db, niche)) if niche else 0
    prov_info = _get_research_provider_info(db)
    return templates.TemplateResponse(
        request=request,
        name="research.html",
        context={
            "active_page": "research",
            "error": None,
            "error_type": None,
            "error_title": None,
            "success": None,
            "niche_val": niche or "",
            "product_count_val": product_count,
            "fresh_val": fresh,
            "cached_count": cached_count,
            "provider_info": prov_info,
            "gemini_status": prov_info.get("gemini_status")
        }
    )



@router.post("/research", response_class=HTMLResponse)
def run_research(
    request: Request,
    niche: str = Form(...),
    product_count: int = Form(10),
    fresh: bool = Form(False),
    db: Session = Depends(get_db)
):
    niche_clean = niche.strip()
    norm_niche = normalize_niche(niche_clean)

    # 1. Deterministic Niche Cache Check (0 AI calls if cached results satisfy count)
    if not fresh:
        cached_prods = get_cached_products_for_niche(db, niche_clean)
        if len(cached_prods) >= product_count:
            logger.info(
                f"Research Cache Hit for niche '{niche_clean}' ({len(cached_prods)} available, "
                f"{product_count} requested). Reusing existing products with 0 LLM calls."
            )
            return RedirectResponse(
                url=f"/products?success=1&count={product_count}&cached=1&niche={niche_clean}",
                status_code=303
            )

    # 2. Invoke AI Provider Manager with strict Low-Consumption Mode (exactly ONE LLM attempt)
    ai_manager = get_ai_manager()
    try:
        products_data = ai_manager.generate_products(niche=niche_clean, count=product_count, db=db)
        if not products_data:
            raise ValueError("Không có sản phẩm nào được tạo ra.")

        # Existing products for this niche to prevent duplicate product names
        existing_names = {
            p.name_vietnamese.strip().lower()
            for p in db.query(Product).all()
            if normalize_niche(p.niche) == norm_niche
        }

        created_products = []
        for p in products_data:
            p_name = p.get("name_vietnamese", "").strip()
            # If not fresh, deduplicate against existing names in this niche
            if p_name.lower() in existing_names and not fresh:
                continue

            pid = get_next_product_id(db)
            prod = Product(
                product_id=pid,
                niche=niche_clean,
                name_vietnamese=p_name or "Chưa đặt tên",
                name_chinese=p.get("name_chinese", ""),
                douyin_keywords=p.get("douyin_keywords", ""),
                content_angle=p.get("content_angle", ""),
                hook=p.get("hook", ""),
                status="RESEARCHED"
            )
            db.add(prod)
            db.flush()
            created_products.append(prod)
            existing_names.add(p_name.lower())

        db.commit()
        count_saved = len(created_products) if created_products else len(products_data)
        return RedirectResponse(
            url=f"/products?success=1&count={count_saved}&niche={niche_clean}",
            status_code=303
        )

    except (AIQuotaExceededError, GeminiQuotaExceededError) as qe:
        db.rollback()
        clean_qe = sanitize_error_message(str(qe))
        prov_info = _get_research_provider_info(db)
        pname = prov_info["provider_name"]
        return templates.TemplateResponse(
            request=request,
            name="research.html",
            context={
                "active_page": "research",
                "error": f"Đã chạm giới hạn hạn mức {pname} (HTTP 429). Research đã dừng và không gửi thêm yêu cầu để bảo vệ hạn ngạch tài khoản. {clean_qe}",
                "error_type": "QUOTA_EXCEEDED",
                "error_title": f"{pname} đã hết hạn mức API / quota",
                "niche_val": niche,
                "product_count_val": product_count,
                "fresh_val": fresh,
                "provider_info": prov_info,
                "gemini_status": prov_info.get("gemini_status")
            },
            status_code=429
        )
    except Exception as e:
        db.rollback()
        err_str = sanitize_error_message(str(e))
        prov_info = _get_research_provider_info(db)
        pname = prov_info["provider_name"]
        error_type = "ERROR"
        error_title = f"Lỗi {pname}"

        if isinstance(e, AIServiceUnavailableError) or "503" in err_str or "quá tải" in err_str.lower() or "overload" in err_str.lower() or "temporarily unavailable" in err_str.lower():
            user_msg = f"{pname} đang tạm thời quá tải (HTTP 503). Research đã dừng để không phát sinh thêm yêu cầu. Bạn có thể thử lại sau vài phút."
            error_type = "BUSY"
            error_title = f"{pname} đang quá tải"
        elif isinstance(e, AIRateLimitError) or "429" in err_str or "rate limit" in err_str.lower() or "quota" in err_str.lower():
            user_msg = f"{pname} đang giới hạn số yêu cầu tạm thời hoặc đã hết quota (HTTP 429). Research đã dừng và không tự thử lại. Vui lòng đợi vài phút hoặc kiểm tra tài khoản."
            error_type = "RATE_LIMITED"
            error_title = f"{pname} đã hết hạn mức API / quota"
        elif isinstance(e, AIAuthenticationError) or "401" in err_str or "unauthorized" in err_str.lower() or "api key" in err_str.lower():
            user_msg = f"Lỗi {pname} API Key (HTTP 401). Kiểm tra lại API Key trong Cài đặt."
            error_type = "AUTH_ERROR"
            error_title = f"Lỗi xác thực {pname}"
        elif isinstance(e, AIPermissionError) or "403" in err_str or "forbidden" in err_str.lower() or "permission" in err_str.lower():
            user_msg = f"{pname} từ chối truy cập (HTTP 403). Kiểm tra lại quyền hạn của API Key hoặc model trong Cài đặt."
            error_type = "AUTH_ERROR"
            error_title = f"{pname} từ chối truy cập"
        elif isinstance(e, AIModelNotFoundError) or "404" in err_str or "model_not_found" in err_str.lower():
            user_msg = f"Không tìm thấy mô hình hoặc mô hình không được hỗ trợ trên {pname}."
            error_type = "ERROR"
            error_title = f"Không tìm thấy mô hình {pname}"
        elif isinstance(e, AITimeoutError) or "timed out" in err_str.lower() or "timeout" in err_str.lower():
            user_msg = f"Yêu cầu tới {pname} bị quá thời gian chờ (Timeout). Research đã dừng và không tự thử lại để tiết kiệm hạn mức."
            error_type = "NETWORK_ERROR"
            error_title = f"{pname} hết thời gian chờ"
        elif isinstance(e, AINetworkError) or "network" in err_str.lower() or "connection" in err_str.lower():
            user_msg = f"Không kết nối được {pname}. Research đã dừng và không tự thử lại. Vui lòng kiểm tra kết nối mạng."
            error_type = "NETWORK_ERROR"
            error_title = f"Không kết nối được {pname}"
        elif isinstance(e, AIInvalidResponseError) or "json" in err_str.lower() or "malformed" in err_str.lower():
            user_msg = f"{pname} trả về dữ liệu không hợp lệ. Hệ thống không tự tạo lại để tránh tốn thêm quota: {err_str}"
            error_type = "ERROR"
            error_title = f"Dữ liệu {pname} không hợp lệ"
        else:
            user_msg = f"Lỗi Research ({pname}): {err_str}"
            error_title = f"Lỗi {pname}"

        return templates.TemplateResponse(
            request=request,
            name="research.html",
            context={
                "active_page": "research",
                "error": user_msg,
                "error_type": error_type,
                "error_title": error_title,
                "niche_val": niche,
                "product_count_val": product_count,
                "fresh_val": fresh,
                "provider_info": prov_info,
                "gemini_status": prov_info.get("gemini_status")
            },
            status_code=400
        )

