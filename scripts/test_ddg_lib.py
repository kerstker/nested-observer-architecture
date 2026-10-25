import json
try:
    from duckduckgo_search import DDGS
    print("duckduckgo_search is installed.")
    try:
        results = DDGS().text("bitcoin price", max_results=3)
        print("Search Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Search failed: {e}")
except ImportError:
    print("duckduckgo_search is NOT installed.")
