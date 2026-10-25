import logging
import json
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class WebSearchTool:
    """
    Level 19: Web Search Tool
    Allows Noa to search the internet using DuckDuckGo (via html.duckduckgo.com for no-api-key usage)
    or Baidu. If every engine fails, the search fails - nothing is made up.
    """
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Referer": "https://www.baidu.com/"
        })
        # Suppress SSL warnings for legacy environments
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Suppress RuntimeWarning from duckduckgo_search about renaming
        import warnings
        try:
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="ddgs")
        except:
            pass

    def search(self, query: str, limit: int = 5) -> str:
        """
        Perform a web search and return a summarized string.
        When every engine fails, says so plainly - it never makes results up.
        """
        result = self.try_search(query, limit)
        if result is None:
            return f"（没有搜到「{query}」：搜索引擎都连不上或没有返回内容）"
        return result

    def try_search(self, query: str, limit: int = 5) -> Optional[str]:
        """
        Real search results only. Returns None when every engine failed, so callers
        can tell "learned something" apart from "found nothing".
        """
        try:
            # 0. Try DuckDuckGo Library first (Most reliable if installed)
            try:
                # Direct import from ddgs which is the new package name
                try:
                    from ddgs import DDGS
                except ImportError:
                    # Fallback to old name just in case
                    from duckduckgo_search import DDGS
                
                logger.info(f"Using ddgs library for: {query}")
                
                # In newer versions of duckduckgo_search, DDGS is a class context manager
                # But proxy setting might be needed if env is weird. 
                # Let's try default first.
                with DDGS() as ddgs:
                    # ddgs.text() returns a generator
                    results = [r for r in ddgs.text(query, max_results=limit)]
                    if results:
                        formatted = f"Search Results for '{query}':\n"
                        for i, r in enumerate(results):
                            formatted += f"{i+1}. [{r.get('title')}]({r.get('href')}): {r.get('body')}\n"
                        return formatted
            except ImportError:
                logger.info("ddgs library not found, falling back to scrapers.")
            except Exception as e:
                # Catching generic Exception to be safe against library changes
                logger.warning(f"ddgs library failed: {e}")

            # 1. Try Baidu first (Scraping - more stable in CN)
            logger.info(f"Searching Baidu for: {query}")
            results = self._baidu_search(query, limit)
            
            # 2. If failed, try DuckDuckGo Manual Scraper
            if not results:
                 logger.info("Baidu failed/empty, trying DDG Manual...")
                 results = self._ddg_search(query, limit)

            if not results:
                # No fabricated fallback: invented "results" would be memorised as knowledge
                logger.warning(f"All search engines failed for: {query}")
                return None
            
            formatted = f"Search Results for '{query}':\n"
            for i, r in enumerate(results):
                formatted += f"{i+1}. [{r['title']}]({r['link']}): {r['snippet']}\n"
            return formatted
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None

    def _baidu_search(self, query: str, limit: int) -> List[Dict[str, str]]:
        """
        Simple scraper for Baidu.
        """
        url = "https://www.baidu.com/s"
        params = {"wd": query}
        
        try:
            # verify=False to bypass SSL issues on old Mac Python envs
            resp = self.session.get(url, params=params, timeout=10, verify=False)
            resp.raise_for_status()
            
            # Basic parsing using regex (Fragile)
            import re
            results = []
            
            # Find result containers (c-container)
            # This is a heuristic.
            snippets = re.findall(r'class="c-abstract">(.*?)</div>', resp.text, re.DOTALL)
            titles = re.findall(r'h3 class="t".*?<a.*?>(.*?)</a>', resp.text, re.DOTALL)
            
            count = min(len(titles), len(snippets), limit)
            for i in range(count):
                # Clean HTML tags
                title_clean = re.sub(r'<.*?>', '', titles[i]).strip()
                snippet_clean = re.sub(r'<.*?>', '', snippets[i]).strip()
                
                results.append({
                    "title": title_clean,
                    "link": "https://www.baidu.com/s?wd=" + query, # Simplified link
                    "snippet": snippet_clean
                })
                
            return results
        except Exception as e:
            logger.warning(f"Baidu Search failed: {e}")
            return []

    def _parse_ddg_html(self, html: str, limit: int) -> List[Dict[str, str]]:
        """
        Very rough parser for DDG HTML.
        """
        results = []
        try:
            import re
            
            # Extract result snippets from DDG HTML (HTML-lite version)
            # Structure usually: <div class="result__body"> ... <a class="result__a">Title</a> ... <a class="result__snippet">Snippet</a>
            
            titles = re.findall(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', html, re.DOTALL)
            snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
            
            # If standard parsing fails, check if we just got a CAPTCHA or empty page
            if not titles and "DOCTYPE" in html[:100]:
                 # This means our regex failed on the structure
                 pass

            count = min(len(titles), len(snippets), limit)
            for i in range(count):
                title_clean = re.sub(r'<.*?>', '', titles[i]).strip()
                snippet_clean = re.sub(r'<.*?>', '', snippets[i]).strip()
                
                results.append({
                    "title": title_clean,
                    "link": "https://duckduckgo.com/?q=...", 
                    "snippet": snippet_clean
                })
                
        except Exception as e:
            logger.warning(f"DDG parsing error: {e}")
            pass
            
        if not results:
            # Parsing found nothing (blocked, CAPTCHA or layout change): report no results
            return []
            
        return results[:limit]

    def _ddg_search(self, query: str, limit: int) -> List[Dict[str, str]]:
        """
        Simple scraper for DuckDuckGo HTML version to avoid API keys.
        """
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query}
        
        try:
            # verify=False to bypass SSL issues
            resp = self.session.post(url, data=data, timeout=10, verify=False)
            
            if resp.status_code == 200:
                results = self._parse_ddg_html(resp.text, limit)
                if results:
                    return results
            
            logger.info("DDG returned no results or blocked.")
            return [] # Let the caller handle fallback
            
        except Exception as e:
            logger.warning(f"DDG Search failed: {e}")
            return []
