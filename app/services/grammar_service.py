# services/grammar_service.py
import httpx
from app.core.settings import settings


async def check_structured_grammar(filename: str, sections: list[dict]):
    """Send structured sections to Colab — get corrections back."""
    timeout = httpx.Timeout(
        connect=5.0,
        read=settings.grammar_api_timeout,
        write=20.0,
        pool=5.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{settings.grammar_api_url}/api/grammar/check-structured",
            json={
                "filename": filename,
                "sections": sections,
            },
        )
        response.raise_for_status()
        return response.json()