from langchain_community.utilities import SearxSearchWrapper
from langsmith import traceable


def deduplicate_and_format_sources(
    search_response, max_tokens_per_source, include_raw_content=True
):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.

    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results

    Returns:
        str: Formatted string with deduplicated sources
    """
    sources_list = search_response
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        source["url"] = source["link"]
        source["content"] = source["snippet"]
        if source["url"] not in unique_sources:
            unique_sources[source["url"]] = source

    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get("raw_content", "")
            if raw_content is None:
                raw_content = ""
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"

    return formatted_text.strip()


def format_sources(search_results):
    """Format search results into a bullet-point list of sources.

    Args:
        search_results (dict): Tavily search response containing results

    Returns:
        str: Formatted string with sources and their URLs
    """
    return "\n".join(
        f"* {source['title']} : {source['url']}" for source in search_results
    )


@traceable
def searxng_search(query, include_raw_content=True, max_results=3):
    """Search the web using the SearxNG API.

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from SearxNG in the formatted string
        max_results (int): Maximum number of results to return

    Returns:
        dict: SearxNG search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""

    searx_search = SearxSearchWrapper(
        searx_host="http://127.0.0.1:8080",
        unsecure=True,
        k=max_results,
        engines=["google"],
    )
    ret = searx_search.results(
        query, num_results=max_results, include_raw_content=include_raw_content
    )
    return ret
