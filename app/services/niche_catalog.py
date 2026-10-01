"""
Local Niche Catalog for Content Factory Product Research.
Provides curated, commercially validated Douyin/TikTok product niches organized across 28 broad categories.

Strict Design Rules:
- 100% local, static in-memory catalog.
- Zero runtime AI generation.
- Zero external network dependencies or web search.
- Safe serialization helpers for templating.
"""
from typing import Dict, List, Optional
import json
import random


# 28 Broad Categories with 8-15 practical, commercial, Douyin-friendly specific niches each
NICHE_CATALOG: Dict[str, List[str]] = {
    "Đồ nhà bếp": [
        "dụng cụ bảo quản thực phẩm",
        "dụng cụ nấu ăn thông minh",
        "đồ vệ sinh nhà bếp",
        "dụng cụ làm bánh",
        "phụ kiện bồn rửa",
        "dụng cụ cắt gọt",
        "đồ tổ chức tủ bếp",
        "đồ tiết kiệm không gian bếp",
        "hộp và túi bảo quản",
        "dụng cụ sơ chế thực phẩm",
        "phụ kiện nồi chiên không dầu",
        "dụng cụ pha chế tại nhà",
    ],
    "Đồ gia dụng": [
        "đồ tiện ích gia đình",
        "thiết bị gia dụng mini",
        "đồ tiết kiệm không gian",
        "dụng cụ vệ sinh đa năng",
        "đồ chống bụi",
        "đồ chống ẩm",
        "phụ kiện máy giặt",
        "phụ kiện máy hút bụi",
        "đồ tổ chức gia đình",
        "sản phẩm tiện ích cho căn hộ nhỏ",
    ],
    "Công nghệ & phụ kiện": [
        "phụ kiện điện thoại",
        "phụ kiện laptop",
        "phụ kiện bàn làm việc",
        "sạc và cáp thông minh",
        "thiết bị Bluetooth mini",
        "đèn setup bàn làm việc",
        "phụ kiện livestream",
        "thiết bị thông minh mini",
        "phụ kiện màn hình",
        "phụ kiện gaming",
        "phụ kiện làm việc tại nhà",
        "thiết bị công nghệ du lịch",
    ],
    "Điện thoại & phụ kiện": [
        "giá đỡ điện thoại",
        "phụ kiện MagSafe",
        "sạc nhanh đa năng",
        "cáp sạc chống đứt",
        "phụ kiện quay video bằng điện thoại",
        "phụ kiện chụp ảnh điện thoại",
        "đồ bảo vệ điện thoại",
        "phụ kiện livestream điện thoại",
        "thiết bị âm thanh cho điện thoại",
    ],
    "Laptop & setup bàn làm việc": [
        "giá đỡ laptop công thái học",
        "bàn phím cơ mini",
        "chuột công thái học",
        "hub USB đa năng",
        "phụ kiện quản lý dây cáp",
        "đèn treo màn hình",
        "phụ kiện công thái học",
        "đồ setup bàn làm việc tối giản",
        "phụ kiện làm việc từ xa",
        "đồ tối ưu không gian bàn",
    ],
    "Làm đẹp": [
        "dụng cụ skincare",
        "dụng cụ chăm sóc tóc",
        "phụ kiện trang điểm",
        "dụng cụ làm nail tại nhà",
        "dụng cụ massage mặt",
        "đồ tổ chức mỹ phẩm",
        "thiết bị làm đẹp mini",
        "phụ kiện tạo kiểu tóc",
        "đồ chăm sóc da tại nhà",
    ],
    "Chăm sóc cá nhân": [
        "dụng cụ vệ sinh cá nhân",
        "đồ chăm sóc răng miệng",
        "dụng cụ massage thư giãn",
        "đồ chăm sóc gót chân",
        "đồ chăm sóc da đầu",
        "phụ kiện phòng tắm cá nhân",
        "thiết bị chăm sóc cá nhân mini",
        "đồ tiện ích sức khỏe hằng ngày",
    ],
    "Mẹ & Bé": [
        "đồ ăn dặm cho bé",
        "dụng cụ ăn uống thông minh cho bé",
        "đồ chơi phát triển kỹ năng",
        "đồ chăm sóc trẻ sơ sinh",
        "đồ an toàn cho trẻ trong nhà",
        "đồ lưu trữ đồ dùng trẻ em",
        "phụ kiện xe đẩy",
        "đồ tiện ích cho mẹ bỉm sữa",
        "đồ dùng khi đưa bé ra ngoài",
        "đồ tổ chức phòng trẻ em",
    ],
    "Thú cưng": [
        "đồ chơi cho mèo",
        "đồ chơi cho chó",
        "dụng cụ vệ sinh thú cưng",
        "bát ăn và khay uống tự động",
        "phụ kiện dắt thú cưng đi dạo",
        "dụng cụ chăm sóc lông và chải lông",
        "đồ tiện ích cho người nuôi mèo",
        "đồ tiện ích cho người nuôi chó",
        "đồ bảo vệ sofa và nội thất khỏi thú cưng",
        "phụ kiện vận chuyển thú cưng",
    ],
    "Ô tô": [
        "phụ kiện nội thất ô tô",
        "đồ vệ sinh ô tô",
        "giá đỡ điện thoại trên ô tô",
        "đồ tổ chức cốp xe gọn gàng",
        "phụ kiện đệm ghế ô tô",
        "đồ tiện ích khi lái xe",
        "đồ bảo vệ nội thất xe",
        "phụ kiện tẩu sạc trên ô tô",
        "đồ khử mùi và làm thơm xe",
        "dụng cụ chăm sóc xe hơi",
    ],
    "Xe máy": [
        "giá đỡ điện thoại xe máy chống rung",
        "phụ kiện áo mưa và đồ đi mưa",
        "bạt phủ và đồ bảo vệ xe",
        "dụng cụ vệ sinh xe máy",
        "túi treo và phụ kiện cốp xe",
        "đồ tiện ích khi chạy xe hằng ngày",
        "phụ kiện đi phượt xe máy",
        "đồ sắp xếp hành lý xe máy",
    ],
    "Phòng ngủ": [
        "đồ tổ chức phòng ngủ",
        "đồ tiện ích đầu giường",
        "phụ kiện gối và giường ngủ",
        "đèn ngủ cảm ứng thông minh",
        "túi chống bụi phòng ngủ",
        "hộp lưu trữ quần áo theo mùa",
        "đồ tiết kiệm không gian phòng ngủ",
        "đồ decor phòng ngủ ấm cúng",
    ],
    "Phòng tắm": [
        "đồ tổ chức phòng tắm",
        "kệ dán tường phòng tắm không khoan",
        "dụng cụ vệ sinh phòng tắm",
        "phụ kiện vòi sen tăng áp",
        "thảm chống trượt nhà tắm",
        "đồ chống ẩm mốc phòng tắm",
        "giỏ lưu trữ nhà tắm",
        "đồ tiện ích bồn cầu",
    ],
    "Lưu trữ & sắp xếp": [
        "hộp lưu trữ trong suốt",
        "kệ mini đa tầng",
        "khay chia ngăn kéo",
        "đồ tổ chức tủ quần áo",
        "đồ chia ngăn tủ đồ lót",
        "đồ tổ chức kệ bếp",
        "đồ tổ chức bàn làm việc",
        "hộp lưu trữ gầm giường",
        "túi hút chân không tiết kiệm không gian",
        "phụ kiện kẹp và quản lý dây",
    ],
    "Vệ sinh nhà cửa": [
        "dụng cụ lau sàn thông minh",
        "chổi và cọ vệ sinh khe nhỏ",
        "cây lau kính cán dài",
        "khăn lau bếp siêu thấm hút",
        "dụng cụ làm sạch bồn cầu",
        "dụng cụ lăn sạch lông thú",
        "bàn chải làm sạch sofa và nệm",
        "bộ dụng cụ làm sạch giày",
        "dụng cụ vệ sinh đa năng cầm tay",
    ],
    "Decor nhà cửa": [
        "đèn led decor đa màu",
        "decor bàn làm việc phong cách tối giản",
        "đồ decor phòng ngủ nhẹ nhàng",
        "decor góc phòng khách hiện đại",
        "phụ kiện móc treo tường nghệ thuật",
        "đồ trang trí phong cách Wabi-sabi",
        "decor căn hộ mini",
        "đồ trang trí theo mùa và lễ hội",
    ],
    "Văn phòng": [
        "phụ kiện sắp xếp bàn làm việc văn phòng",
        "khay tài liệu và kẹp hồ sơ thông minh",
        "dụng cụ văn phòng phẩm thông minh",
        "đệm đỡ lưng chống mỏi khi ngồi làm",
        "phụ kiện laptop văn phòng",
        "hộp giấu ổ cắm và quản lý dây",
        "đèn bàn làm việc chống lóa",
        "đồ tiện ích cho người làm việc từ xa",
    ],
    "Học tập": [
        "dụng cụ học tập thông minh",
        "phụ kiện bàn học gọn gàng",
        "giá đỡ và kệ sắp xếp sách vở",
        "đèn học bảo vệ mắt",
        "đồ dùng tiện ích cho sinh viên",
        "giá đỡ tablet và iPad học online",
        "bút và dụng cụ ghi chú sáng tạo",
        "đồ tối ưu góc học tập nhỏ",
    ],
    "Du lịch": [
        "set túi tổ chức hành lý vali",
        "phụ kiện dây đai và khóa vali",
        "gối cổ chữ U và bịt mắt máy bay",
        "set chai lọ chiết mỹ phẩm du lịch",
        "túi chống nước điện thoại du lịch",
        "thẻ định vị chống thất lạc hành lý",
        "phụ kiện đi biển gọn nhẹ",
        "đồ dùng cá nhân du lịch dùng 1 lần",
        "bàn ủi du lịch mini",
        "ổ cắm chuyển đổi quốc tế đa năng",
    ],
    "Thể thao & fitness": [
        "dây kháng lực và phụ kiện tập gym",
        "dụng cụ tập luyện tại nhà nhỏ gọn",
        "thảm và phụ kiện tập yoga",
        "đai đeo điện thoại chạy bộ",
        "con lăn và súng massage phục hồi cơ",
        "bình nước thể thao có vạch chia",
        "con lăn tập bụng bánh kép",
        "dụng cụ nhảy dây không dây",
        "túi đựng đồ tập gym chuyên dụng",
    ],
    "Outdoor / dã ngoại": [
        "dụng cụ cắm trại mini gấp gọn",
        "thảm và phụ kiện picnic",
        "đèn lều và đèn chiếu sáng ngoài trời",
        "bộ nồi chảo nấu ăn dã ngoại",
        "móc treo và phụ kiện balo dã ngoại",
        "bạt che và đồ chống mưa ngoài trời",
        "dụng cụ sinh tồn mini bỏ túi",
        "ghế xếp du lịch dã ngoại siêu nhẹ",
        "bình giữ nhiệt dã ngoại dung tích lớn",
    ],
    "Thời trang & phụ kiện": [
        "phụ kiện thời trang nữ hot trend",
        "phụ kiện thời trang nam thanh lịch",
        "túi xách mini đeo chéo",
        "kẹp tóc và phụ kiện tạo kiểu tóc",
        "kính râm thời trang chống UV",
        "hộp đựng và phụ kiện trang sức",
        "khay tổ chức phụ kiện thời trang",
        "băng dính dán áo và phụ kiện quần áo",
        "dụng cụ cắt xơ vải quần áo",
    ],
    "Giày dép & phụ kiện": [
        "bộ bọt vệ sinh giày sneaker",
        "hộp bảo quản và trưng bày giày",
        "miếng lót giày êm chân tăng chiều cao",
        "miếng dán gót chân chống phồng rộp",
        "kệ để giày gấp gọn tiết kiệm diện tích",
        "bàn chải chuyên dụng làm sạch da lộn",
        "dây giày co giãn thông minh không cần buộc",
        "cây giữ form chống gãy mũi giày",
    ],
    "Đồ chơi": [
        "đồ chơi lắp ráp sáng tạo",
        "đồ chơi giáo dục STEM cho trẻ",
        "đồ chơi vận động tương tác trong nhà",
        "đồ chơi mini xả stress bàn làm việc",
        "board game giải trí gia đình",
        "đồ chơi phát triển tư duy logic",
        "đồ chơi búp bê và mô hình nhân vật",
        "đồ chơi tương tác ngoài trời",
    ],
    "Quà tặng": [
        "quà tặng sinh nhật độc đáo",
        "quà tặng tinh tế cho nam",
        "quà tặng xinh xắn cho nữ",
        "quà tặng đồng nghiệp văn phòng",
        "quà tặng cặp đôi kỷ niệm",
        "quà tặng tiện ích thực tế",
        "quà tặng công nghệ sáng tạo",
        "quà tặng bất ngờ giá bình dân",
        "hộp quà đóng gói sẵn tiện lợi",
    ],
    "DIY / dụng cụ sửa chữa": [
        "bộ tua vít sửa chữa điện thoại mini",
        "thước đo dây và thước laser mini",
        "bộ dụng cụ sửa chữa gia đình đa năng",
        "bộ mũi khoan đa vật liệu",
        "kìm và dụng cụ cầm tay nhỏ gọn",
        "bộ hàn và phụ kiện sửa điện tử",
        "hộp đựng ốc vít và phụ kiện linh kiện",
        "súng bắn keo nến mini",
        "băng keo chống thấm đa năng",
    ],
    "Sân vườn": [
        "bộ dụng cụ làm vườn ban công mini",
        "vòi xịt và hệ thống tưới cây tự động",
        "phụ kiện chậu cây treo tường thông minh",
        "đèn năng lượng mặt trời cắm sân vườn",
        "dụng cụ kéo cắt tỉa cành cây cảnh",
        "khay ươm hạt và trồng rau mầm tại nhà",
        "dụng cụ xới đất mini",
        "phụ kiện decor tiểu cảnh ban công",
    ],
    "Đồ tiện ích hằng ngày": [
        "đồ tiện ích mini bỏ túi",
        "sản phẩm giải quyết vấn đề hằng ngày",
        "đồ gia dụng tiết kiệm thời gian",
        "sản phẩm tiết kiệm không gian nhà ở",
        "đồ tiện ích mang theo khi ra ngoài",
        "sản phẩm đa năng 2-trong-1",
        "đồ dùng tối ưu cho căn hộ nhỏ",
        "sản phẩm tiện ích cho người bận rộn",
        "đồ tiện lợi giá bình dân",
    ],
}


def get_broad_categories() -> List[str]:
    """Return all 28 broad category names in catalog order."""
    return list(NICHE_CATALOG.keys())


def get_niches_for_category(category: str) -> List[str]:
    """Return list of specific niches for a broad category (empty if not found)."""
    return list(NICHE_CATALOG.get(category, []))


def get_all_niches() -> List[str]:
    """Return flattened list of all specific niches across all categories."""
    all_n: List[str] = []
    for niches in NICHE_CATALOG.values():
        all_n.extend(niches)
    return all_n


def get_random_niche_sample(sample_size: int = 8, exclude: Optional[List[str]] = None) -> List[str]:
    """
    Return a pseudo-random sample of specific niches from the entire catalog.
    Excludes any items specified in `exclude` to avoid immediate duplicates.
    Deterministic fallback if sample_size exceeds available pool.
    """
    all_niches = get_all_niches()
    if exclude:
        ex_set = set(exclude)
        pool = [n for n in all_niches if n not in ex_set]
        if len(pool) < sample_size:
            pool = all_niches
    else:
        pool = all_niches

    if not pool:
        return []

    sample_k = min(sample_size, len(pool))
    return random.sample(pool, sample_k)


def get_catalog_dict() -> Dict[str, List[str]]:
    """Return a defensive copy of the full catalog dictionary."""
    return {cat: list(niches) for cat, niches in NICHE_CATALOG.items()}
