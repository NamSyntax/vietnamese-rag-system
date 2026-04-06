from src.generation.generator import RAGGenerator


def test_build_messages_formatting():
    generator = RAGGenerator(max_context_length=1000)
    query = "Luật bảo hiểm là gì?"
    contexts = [{"content": "Bảo hiểm là biện pháp chia sẻ rủi ro."}]

    messages = generator._build_messages(query, contexts)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert (
        "Tuân thủ các quy tắc" in messages[0]["content"].lower()
        or "quy tắc" in messages[0]["content"].lower()
    )

    assert messages[1]["role"] == "user"
    assert "Luật bảo hiểm là gì?" in messages[1]["content"]
    assert "Bảo hiểm là biện pháp chia sẻ rủi ro." in messages[1]["content"]


def test_build_messages_truncation_limit():
    generator = RAGGenerator(max_context_length=150)  # tight limit
    query = "Hỏi điều khoản?"

    contexts = [
        {"content": "a" * 100},  # Length ~ 100 inside template
        {"content": "b" * 100},  # Should be truncated explicitly because length exceeds 150
    ]

    messages = generator._build_messages(query, contexts)
    user_content = messages[1]["content"]

    # First context block 'a' should be there
    assert "a" * 100 in user_content
    # Second context block 'b' should be dropped to respect context limit
    assert "b" * 100 not in user_content
