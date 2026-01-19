"""
Vector-Based Tool Search

Uses sentence embeddings for semantic similarity search.
This gives much better results than keyword matching because it understands meaning.

For example:
- Query "current time" matches "get_datetime" even without word overlap
- Query "send a message" matches "post_to_slack", "send_email", etc.

Requirements:
    pip install sentence-transformers numpy faiss-cpu anthropic
"""

import json
import numpy as np
from anthropic import Anthropic


try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer


class VectorToolSearcher:
    """
    Semantic search over tools using sentence embeddings.
    
    Supports two backends:
    - NumPy: Simple cosine similarity (good for <1000 tools)
    - FAISS: Approximate nearest neighbor (good for 1000+ tools)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_faiss: bool = True):
        """
        Args:
            model_name: Sentence transformer model. Options:
                - "all-MiniLM-L6-v2" (fast, 384 dims)
                - "all-mpnet-base-v2" (better quality, 768 dims)
                - "multi-qa-MiniLM-L6-cos-v1" (optimized for Q&A)
            use_faiss: Use FAISS for faster search with large tool sets
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        self.tools: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.faiss_index = None
    
    def index(self, tools: list[dict]):
        """
        Index tools for semantic search.
        
        Args:
            tools: List of {"name": str, "description": str, ...}
        """
        self.tools = tools
        
        # Create searchable text from name + description
        texts = [
            f"{t['name'].replace('_', ' ')}: {t['description']}"
            for t in tools
        ]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(tools)} tools...")
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
            show_progress_bar=True
        )
        
        # Build FAISS index if requested
        if self.use_faiss:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
            self.faiss_index.add(self.embeddings.astype(np.float32))
            print(f"Built FAISS index with {len(tools)} vectors")
        
        print(f"✓ Indexed {len(tools)} tools")
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for tools matching the query.
        
        Returns list of {"name": str, "description": str, "score": float}
        """
        # Embed query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]
        
        if self.use_faiss:
            # FAISS search
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                top_k
            )
            results = [
                {**self.tools[idx], "score": float(score)}
                for score, idx in zip(scores[0], indices[0])
                if score > 0.1  # Filter low scores
            ]
        else:
            # NumPy cosine similarity
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = [
                {**self.tools[idx], "score": float(similarities[idx])}
                for idx in top_indices
                if similarities[idx] > 0.1
            ]
        
        return results


class VectorToolSearchAdvisor:
    """Tool Search Tool pattern with vector-based search."""
    
    def __init__(
        self,
        client: Anthropic,
        searcher: VectorToolSearcher,
        model: str = "claude-sonnet-4-20250514"
    ):
        self.client = client
        self.searcher = searcher
        self.model = model
        
        self.tool_registry: dict[str, dict] = {}
        self.active_tools: set[str] = set()
        
        # Stats
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.requests = 0
    
    def register_tools(self, tools: list[dict]):
        """
        Register tools and build search index.
        
        Each tool should have:
            - name: str
            - description: str  
            - parameters: dict (JSON Schema)
            - function: callable
        """
        self.tool_registry = {t["name"]: t for t in tools}
        
        # Index for search (just name + description)
        self.searcher.index([
            {"name": t["name"], "description": t["description"]}
            for t in tools
        ])
    
    def _build_tools_payload(self) -> list[dict]:
        """Build tools array for API call."""
        tools = [{
            "name": "search_tools",
            "description": (
                "Discover available tools by describing what you need. "
                "ALWAYS call this first before using any other tool. "
                "Returns matching tools with their descriptions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Describe the capability you're looking for"
                    }
                },
                "required": ["query"]
            }
        }]
        
        for name in self.active_tools:
            tool = self.tool_registry[name]
            tools.append({
                "name": name,
                "description": tool["description"],
                "input_schema": {
                    "type": "object",
                    "properties": tool["parameters"].get("properties", {}),
                    "required": tool["parameters"].get("required", [])
                }
            })
        
        return tools
    
    def _execute(self, name: str, inputs: dict) -> str:
        if name == "search_tools":
            results = self.searcher.search(inputs.get("query", ""), top_k=5)
            
            for r in results:
                if r["name"] not in self.active_tools:
                    self.active_tools.add(r["name"])
                    print(f"    → Discovered: {r['name']} (score: {r['score']:.2f})")
            
            return json.dumps({
                "found_tools": [
                    {"name": r["name"], "description": r["description"]}
                    for r in results
                ]
            })
        
        elif name in self.active_tools:
            func = self.tool_registry[name]["function"]
            try:
                return json.dumps({"result": func(**inputs)})
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        return json.dumps({"error": f"Tool '{name}' not found. Search first."})
    
    def chat(self, message: str, max_turns: int = 10) -> str:
        """Process message with dynamic tool discovery."""
        
        self.active_tools = set()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.requests = 0
        
        messages = [{"role": "user", "content": message}]
        
        for turn in range(max_turns):
            print(f"\n[Turn {turn + 1}] Active: {list(self.active_tools)}")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=(
                    "You have access to many tools through a search interface. "
                    "Use search_tools to discover capabilities before using them."
                ),
                tools=self._build_tools_payload(),
                messages=messages
            )
            
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.requests += 1
            
            print(f"  Tokens: {response.usage.input_tokens}↓ {response.usage.output_tokens}↑")
            
            if response.stop_reason == "end_turn":
                print(f"\n📊 Total: {self.requests} requests, "
                      f"{self.total_input_tokens} input, "
                      f"{self.total_output_tokens} output tokens")
                
                for block in response.content:
                    if block.type == "text":
                        return block.text
                return ""
            
            # Process tool calls
            assistant_content = []
            tool_results = []
            
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    print(f"  🔧 {block.name}({json.dumps(block.input)[:50]}...)")
                    
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
                    
                    result = self._execute(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        
        return "Max turns reached"


# =============================================================================
# DEMO
# =============================================================================

def create_sample_tools():
    """Create a diverse set of sample tools."""
    return [
        {
            "name": "get_current_weather",
            "description": "Get real-time weather conditions including temperature, humidity, and forecast for any city",
            "parameters": {
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            },
            "function": lambda city: {"city": city, "temp_c": 18, "conditions": "Partly cloudy"}
        },
        {
            "name": "get_datetime",
            "description": "Get the current date and time for a specific timezone",
            "parameters": {
                "properties": {"timezone": {"type": "string"}},
                "required": ["timezone"]
            },
            "function": lambda timezone: {"timezone": timezone, "datetime": "2025-01-05T14:30:00"}
        },
        {
            "name": "find_retail_stores",
            "description": "Find shopping stores like clothing, electronics, or grocery stores in a location",
            "parameters": {
                "properties": {
                    "location": {"type": "string"},
                    "category": {"type": "string"}
                },
                "required": ["location"]
            },
            "function": lambda location, category="all": {
                "stores": ["H&M", "Zara", "Primark"],
                "location": location
            }
        },
        {
            "name": "send_email",
            "description": "Send an email message to one or more recipients",
            "parameters": {
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            },
            "function": lambda to, subject, body: {"status": "sent"}
        },
        {
            "name": "query_database",
            "description": "Execute SQL queries against the application database",
            "parameters": {
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"]
            },
            "function": lambda sql: {"rows": [], "count": 0}
        },
        {
            "name": "create_calendar_event",
            "description": "Schedule a new meeting or event on the calendar",
            "parameters": {
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"}
                },
                "required": ["title", "start", "end"]
            },
            "function": lambda title, start, end: {"event_id": "evt_123"}
        },
        {
            "name": "post_slack_message",
            "description": "Send a message to a Slack channel or direct message a user",
            "parameters": {
                "properties": {
                    "channel": {"type": "string"},
                    "text": {"type": "string"}
                },
                "required": ["channel", "text"]
            },
            "function": lambda channel, text: {"ok": True}
        },
        {
            "name": "translate_text",
            "description": "Translate text between languages",
            "parameters": {
                "properties": {
                    "text": {"type": "string"},
                    "target_language": {"type": "string"}
                },
                "required": ["text", "target_language"]
            },
            "function": lambda text, target_language: {"translated": f"[{target_language}] {text}"}
        },
        # Add more dummy tools to simulate large tool library
        *[
            {
                "name": f"internal_tool_{i}",
                "description": f"Internal enterprise tool for department {i} workflows",
                "parameters": {"properties": {}},
                "function": lambda: {}
            }
            for i in range(30)
        ]
    ]


def main():
    print("🚀 Vector Tool Search Demo\n")
    
    # Initialize
    client = Anthropic()
    searcher = VectorToolSearcher(model_name="all-MiniLM-L6-v2")
    advisor = VectorToolSearchAdvisor(client, searcher)
    
    # Register tools
    tools = create_sample_tools()
    advisor.register_tools(tools)
    
    print(f"\n📚 Registered {len(tools)} tools")
    
    # Test semantic search directly
    print("\n" + "="*60)
    print("Testing semantic search:")
    print("="*60)
    
    test_queries = [
        "what time is it",
        "clothing shops nearby",
        "message my team",
        "check the forecast"
    ]
    
    for q in test_queries:
        results = searcher.search(q, top_k=3)
        print(f"\n'{q}' →")
        for r in results:
            print(f"  {r['name']} ({r['score']:.2f})")
    
    # Full conversation test
    print("\n" + "="*60)
    print("Full conversation test:")
    print("="*60)
    print("USER: What's the weather in Amsterdam and what time is it there?")
    print("      Also find me some clothing stores.")
    print("="*60)
    
    response = advisor.chat(
        "What's the weather in Amsterdam and what time is it there? "
        "Also find me some clothing stores I could visit."
    )
    
    print("\n" + "="*60)
    print("ASSISTANT:")
    print("="*60)
    print(response)


if __name__ == "__main__":
    main()