# src/generation/generator.py
import httpx
import json
import logging

logger = logging.getLogger(__name__)

class RAGGenerator:
    def __init__(
        self, 
        model_name: str = "qwen2.5:7b-instruct", 
        api_url: str = "http://localhost:11434/api/chat",
        bot_name: str = "VietRAG Bot",
        creator_name: str = "NamSyntax",
        temperature: float = 0.1,
        max_context_length: int = 25000
    ):
        self.url = api_url
        self.model_name = model_name
        self.bot_name = bot_name
        self.creator_name = creator_name
        self.temperature = temperature
        self.max_context_length = max_context_length

    def _build_messages(self, query: str, contexts: list) -> list:
        # 1. System Instruction: Định hình persona và quy tắc gắt gao
        system_instruction = (
            f"Bạn tên là {self.bot_name}, một trợ lý AI thông minh chuyên giải đáp tài liệu, "
            f"được phát triển bởi {self.creator_name}.\n\n"
            "Nhiệm vụ của bạn là trả lời câu hỏi dựa HOÀN TOÀN vào phần <context> được cung cấp.\n"
            "HÃY TUÂN THỦ CÁC QUY TẮC SAU:\n"
            "1. Chỉ sử dụng thông tin trong <context>. Tuyệt đối không dùng kiến thức bên ngoài.\n"
            "2. Trả lời chi tiết, chính xác, lịch sự và dễ hiểu bằng tiếng Việt.\n"
            "3. Nếu <context> KHÔNG chứa thông tin liên quan, hãy trả lời chính xác câu sau: "
            "'Dựa trên tài liệu hiện tại, tôi không tìm thấy thông tin để trả lời câu hỏi này.'\n"
            "4. Không tự bịa đặt thông tin (No hallucination)."
        )

        # chống tràn context (Overflow Protection)
        # Sử dụng XML tags để LLM phân tách rõ ràng dữ liệu
        context_parts = []
        current_length = 0
        
        for i, c in enumerate(contexts):
            chunk_text = f"Tài liệu {i+1}:\n{c['content']}"
            if current_length + len(chunk_text) > self.max_context_length:
                logger.warning("context is too long, truncating to fit the limit.")
                break
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
            
        context_text = "\n\n".join(context_parts)

        # 3. User Content format
        user_content = (
            f"<context>\n{context_text}\n</context>\n\n"
            f"Câu hỏi: {query}"
        )

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    async def generate_stream(self, query: str, contexts: list):
        messages = self._build_messages(query, contexts)
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.95,
                "num_predict": -1, 
                "num_ctx": 8192
            }
        }

        try:
            # timeout 60-120s đảm bảo phản hồi
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, read=120.0)) as client:
                async with client.stream("POST", self.url, json=payload) as response:
                    
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                if "message" in chunk and "content" in chunk["message"]:
                                    content = chunk["message"]["content"]
                                    # chỉ yield nếu có nội dung thực tế
                                    if content: 
                                        yield content
                                        
                                if chunk.get("done"):
                                    break
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Error parsing JSON from LLM chunk: {line}")
                                continue
                                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP Error {e.response.status_code} from Ollama")
            # return lỗi ra app.py thành {"type": "error"}
            raise RuntimeError(f"Server AI trả về lỗi {e.response.status_code}")
            
        except httpx.RequestError as e:
            logger.error(f"Request Error from Ollama: {e}")
            raise ConnectionError("Không thể kết nối tới LLM (Ollama).")