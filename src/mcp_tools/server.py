from mcp.server.fastmcp import FastMCP

from .paper_tools import (
    search as search_papers,
    read as read_paper,
    keyword_grep as grep_papers,
)

mcp = FastMCP("rag-paper-tools")


@mcp.tool()
def search(query: str, limit: int = 5) -> list[dict]:
    """Search paper chunks and return matching evidence."""
    return search_papers(query, limit)


@mcp.tool()
def read(paper_id: str, section: str = "") -> str:
    """Read a complete paper or a particular section."""
    return read_paper(paper_id, section)


@mcp.tool()
def keyword_grep(keyword: str, limit: int = 20) -> list[dict]:
    """Find literal keyword matches in the corpus."""
    return grep_papers(keyword, limit)


if __name__ == "__main__":
    mcp.run(transport="stdio")