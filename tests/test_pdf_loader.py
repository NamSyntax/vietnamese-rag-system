from src.ingestion.pdf_loader import PDFIngestionPipeline


def test_is_noise_filter():
    pipeline = PDFIngestionPipeline()

    # discard tiny metadata headers
    assert pipeline._is_noise("Header ngắn") is True
    assert pipeline._is_noise("Tài liệu chính thức") is True  # 19 chars -> True

    # discard digits-only markers
    assert pipeline._is_noise("  123  ") is True

    # accept logical sentences
    assert (
        pipeline._is_noise(
            "Đây là một câu văn hoàn chỉnh và dài hơn hai mươi ký tự để đảm bảo mang ý nghĩa thông tin."
        )
        is False
    )


def test_normalize_text():
    pipeline = PDFIngestionPipeline()

    bad_list_text = "o Mục một\n• Mục hai\n➢ Mục ba\n"

    normalized = pipeline._normalize_text(bad_list_text)
    assert "- Mục một" in normalized
    assert "- Mục hai" in normalized
    assert "- Mục ba" in normalized
    assert "o " not in normalized
    assert "• " not in normalized


def test_structural_chunking():
    pipeline = PDFIngestionPipeline(max_chunk_length=100, chunk_overlap=20)

    # structural lists buffer
    text = (
        "Khái quát hệ thống luật.\n"
        "Luật có các đặc điểm cơ bản sau:\n"
        "- Tính quy phạm chuẩn mực chung.\n"  # structure flag here
        "- Tính bắt buộc chung.\n"
        "- Tính cưỡng chế nhà nước.\n\n"
        "Luật cũng cần đảm bảo sự ổn định trong một khoảng thời gian dài."
    )

    chunks = pipeline._hybrid_structural_chunking(text)

    assert len(chunks) > 0  # prevent empty parsing
    # assert cross overlap block
    assert any("nhà nước" in chunk for chunk in chunks)
