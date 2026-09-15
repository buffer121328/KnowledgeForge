"""Compatible ASGI entrypoint for the KnowledgeForge API."""

from api.app import app


if __name__ == "__main__":
    import uvicorn

    from shared.config import settings

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
