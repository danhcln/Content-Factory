from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path

from app.database import get_db

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

from fastapi import Form
from fastapi.responses import RedirectResponse
from app.models import Product
from app.services.gemini_service import GeminiService, get_next_product_id, GeminiQuotaExceededError
from app.services.gemini_status import GeminiStatusTracker

router = APIRouter(tags=["research"])


def _get_gemini_status(db: Session) -> dict:
    """Load current local Gemini status for display — no network calls."""
    return GeminiStatusTracker.get_status(db=db)


@router.get("/research", response_class=HTMLResponse)
def get_research_page(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        request=request,
        name="research.html",
        context={
            "active_page": "research",
            "error": None,
            "success": None,
            "gemini_status": _get_gemini_status(db)
        }
    )


@router.post("/research", response_class=HTMLResponse)
def run_research(
    request: Request,
    niche: str = Form(...),
    product_count: int = Form(10),
    db: Session = Depends(get_db)
):
    gemini = GeminiService()
    try:
        products_data = gemini.generate_products(niche=niche, count=product_count, db=db)
        if not products_data:
            raise ValueError("Không có sản phẩm nào được tạo ra.")

        created_products = []
        for p in products_data:
            pid = get_next_product_id(db)
            prod = Product(
                product_id=pid,
                niche=niche.strip(),
                name_vietnamese=p.get("name_vietnamese", "Chưa đặt tên"),
                name_chinese=p.get("name_chinese", ""),
                douyin_keywords=p.get("douyin_keywords", ""),
                content_angle=p.get("content_angle", ""),
                hook=p.get("hook", ""),
                status="RESEARCHED"
            )
            db.add(prod)
            db.flush()
            created_products.append(prod)

        db.commit()
        return RedirectResponse(url=f"/products?success=1&count={len(created_products)}", status_code=303)
    except GeminiQuotaExceededError:
        db.rollback()
        return templates.TemplateResponse(
            request=request,
            name="research.html",
            context={
                "active_page": "research",
                "error": "Đã hết hạn mức Gemini hôm nay. Chờ hạn mức được làm mới hoặc sử dụng project/tier có hạn mức phù hợp.",
                "error_type": "QUOTA_EXCEEDED",
                "niche_val": niche,
                "product_count_val": product_count,
                "gemini_status": _get_gemini_status(db)
            },
            status_code=429
        )
    except Exception as e:
        db.rollback()
        err_str = str(e)
        # Map error messages to user-friendly Vietnamese
        error_type = "ERROR"
        if "503" in err_str or "quá tải" in err_str.lower() or "overload" in err_str.lower():
            user_msg = "Gemini đang quá tải tạm thời. Hệ thống đã thử lại nhưng chưa thành công. Hãy thử lại sau vài phút."
            error_type = "BUSY"
        elif "429" in err_str or "Rate Limit" in err_str:
            user_msg = "Gemini đang giới hạn số yêu cầu tạm thời. Hãy đợi vài giây rồi thử lại."
            error_type = "RATE_LIMITED"
        elif "401" in err_str or "Unauthorized" in err_str or "API Key" in err_str:
            user_msg = "Lỗi Gemini API Key. Kiểm tra lại API Key trong Cài đặt."
            error_type = "AUTH_ERROR"
        elif "403" in err_str or "Forbidden" in err_str:
            user_msg = "Gemini từ chối truy cập. Kiểm tra lại quyền hạn của API Key trong Cài đặt."
            error_type = "AUTH_ERROR"
        elif "timed out" in err_str.lower() or "timeout" in err_str.lower():
            user_msg = "Gemini không phản hồi (timeout). Kiểm tra kết nối mạng và thử lại."
            error_type = "NETWORK_ERROR"
        elif "network" in err_str.lower() or "connection" in err_str.lower():
            user_msg = "Không kết nối được Gemini. Kiểm tra kết nối mạng và thử lại."
            error_type = "NETWORK_ERROR"
        else:
            user_msg = err_str

        return templates.TemplateResponse(
            request=request,
            name="research.html",
            context={
                "active_page": "research",
                "error": user_msg,
                "error_type": error_type,
                "niche_val": niche,
                "product_count_val": product_count,
                "gemini_status": _get_gemini_status(db)
            },
            status_code=400
        )
