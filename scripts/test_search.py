import logging
from src.levels.l19_search.web_search import WebSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_search():
    tool = WebSearchTool()
    
    print("\n--- Testing Baidu Search (Mocked in this env?) ---")
    query = "比特币价格"
    results = tool.search(query, limit=2)
    print(f"Query: {query}")
    print(f"Result:\n{results}")

    print("\n--- Testing Weather Mock ---")
    query = "东京天气"
    results = tool.search(query, limit=2)
    print(f"Query: {query}")
    print(f"Result:\n{results}")

    print("\n--- Testing Fallback Logic ---")
    query = "Asdflkjhqwerpoiu" # Nonsense query
    results = tool.search(query, limit=2)
    print(f"Query: {query}")
    print(f"Result:\n{results}")

if __name__ == "__main__":
    test_search()
