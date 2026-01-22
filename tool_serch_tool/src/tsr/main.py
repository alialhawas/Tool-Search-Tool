"""
Tool Search Tool Pattern - Python Implementation

This implements dynamic tool discovery for LLM agents, achieving significant
token savings when working with large tool libraries (20+ tools).

Based on the pattern described by Anthropic:
https://www.anthropic.com/engineering/advanced-tool-use
"""

import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False


@dataclass
class Tool:
    """Represents a callable tool with its schema."""
    name: str
    description: str
    parameters: dict 
    function: Callable[..., Any]
    
    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic's tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters.get("properties", {}),
                "required": self.parameters.get("required", [])
            }
        }
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        return self.function(**kwargs)

class ToolSearcher(ABC):
    """Abstract base class for tool search strategies."""
    
    @abstractmethod
    def index(self, tools: list[Tool]) -> None:
        """Index all tools for searching."""
        pass
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[Tool]:
        """Search for tools matching the query."""
        pass



class KeywordToolSearcher(ToolSearcher):
    """
    Simple keyword-based tool search using TF-IDF-like scoring.
    No external dependencies required.
    """
    
    def __init__(self):
        self.tools: list[Tool] = []
        self.tool_tokens: dict[str, set[str]] = {}  # tool_name -> tokens
    
    def _tokenize(self, text: str) -> set[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        import re
        tokens = re.findall(r'\w+', text.lower())
        return set(tokens)
    
    def index(self, tools: list[Tool]) -> None:
        """Index tools by their name and description tokens."""
        self.tools = tools
        self.tool_tokens = {}
        
        for tool in tools:
            # Combine name and description for indexing
            text = f"{tool.name} {tool.description}"
            self.tool_tokens[tool.name] = self._tokenize(text)
        
        print(f"[KeywordSearcher] Indexed {len(tools)} tools")
    
    def search(self, query: str, max_results: int = 5) -> list[Tool]:
        """Search tools by keyword overlap."""
        query_tokens = self._tokenize(query)
        
        scores = []
        for tool in self.tools:
            tool_tokens = self.tool_tokens[tool.name]
            # Score = number of matching tokens
            overlap = len(query_tokens & tool_tokens)
            if overlap > 0:
                # Normalize by query length to favor more specific matches
                score = overlap / len(query_tokens)
                scores.append((score, tool))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = [tool for _, tool in scores[:max_results]]
        print(f"[KeywordSearcher] Query '{query}' -> {[t.name for t in results]}")
        return results


class VectorToolSearcher(ToolSearcher):
    """
    Semantic search using sentence embeddings.
    Requires: pip install numpy sentence-transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not VECTOR_SEARCH_AVAILABLE:
            raise ImportError(
                "Vector search requires numpy and sentence-transformers. "
                "Install with: pip install numpy sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self.tools: list[Tool] = []
        self.embeddings: np.ndarray | None = None
    
    def index(self, tools: list[Tool]) -> None:
        """Create embeddings for all tools."""
        self.tools = tools
        
        # Create text representations for embedding
        texts = [f"{t.name}: {t.description}" for t in tools]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        print(f"[VectorSearcher] Indexed {len(tools)} tools with embeddings")
    
    def search(self, query: str, max_results: int = 5) -> list[Tool]:
        """Search tools by semantic similarity."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:max_results]
        
        results = [self.tools[i] for i in top_indices if similarities[i] > 0.35]
        print(f"[VectorSearcher] Query '{query}' -> {[t.name for t in results]}")
        return results


class ToolSearchToolAdvisor:
    """
    Implements the Tool Search Tool pattern for dynamic tool discovery.
    
    Instead of sending all tools to the LLM, it:
    1. Sends only a 'search_tools' tool initially
    2. When LLM searches for tools, discovers relevant ones
    3. Injects discovered tools into subsequent calls
    4. Executes tool calls and loops until complete
    """
    
    def __init__(
        self,
        client: Anthropic,
        tool_searcher: ToolSearcher,
        model: str = "claude-sonnet-4-20250514",
        max_search_results: int = 5,
        max_iterations: int = 10
    ):
        self.client = client
        self.tool_searcher = tool_searcher
        self.model = model
        self.max_search_results = max_search_results
        self.max_iterations = max_iterations
        
        self.all_tools: dict[str, Tool] = {}  # All registered tools
        self.discovered_tools: dict[str, Tool] = {}  # Currently discovered tools
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
    
    def register_tools(self, tools: list[Tool]) -> None:
        """Register and index all available tools."""
        self.all_tools = {t.name: t for t in tools}
        self.tool_searcher.index(tools)
        print(f"[Advisor] Registered {len(tools)} tools")
    
    def _get_search_tool_schema(self) -> dict:
        """The meta-tool that searches for other tools."""
        return {
            "name": "search_tools",
            "description": (
                "Search for available tools by describing what capability you need. "
                "Use this to discover tools before calling them. "
                "Example queries: 'weather forecast', 'send email', 'database query'"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the capability you're looking for"
                    }
                },
                "required": ["query"]
            }
        }
    
    def _get_current_tools(self) -> list[dict]:
        """Get tool schemas: search_tool + any discovered tools."""
        tools = [self._get_search_tool_schema()]
        for tool in self.discovered_tools.values():
            tools.append(tool.to_anthropic_schema())
        return tools
    
    def _handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result."""
        
        if tool_name == "search_tools":
            # Meta-tool: search for tools
            query = tool_input.get("query", "")
            found_tools = self.tool_searcher.search(query, self.max_search_results)
            
            # Add discovered tools to our active set
            for tool in found_tools:
                if tool.name not in self.discovered_tools:
                    self.discovered_tools[tool.name] = tool
                    print(f"[Advisor] Discovered tool: {tool.name}")
            
            # Return tool names and descriptions
            return json.dumps({
                "found_tools": [
                    {"name": t.name, "description": t.description}
                    for t in found_tools
                ]
            })
        
        elif tool_name in self.discovered_tools:
            # Execute a discovered tool
            tool = self.discovered_tools[tool_name]
            try:
                result = tool.execute(**tool_input)
                print(f"[Advisor] Executed {tool_name} -> {result}")
                return json.dumps({"result": result})
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        else:
            return json.dumps({
                "error": f"Tool '{tool_name}' not found. Use search_tools to discover available tools."
            })
    
    def chat(self, user_message: str, system_prompt: str | None = None) -> str:
        """
        Process a user message with dynamic tool discovery.
        
        Returns the final assistant response.
        """
        # Reset state for new conversation
        self.discovered_tools = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        
        messages = [{"role": "user", "content": user_message}]
        
        default_system = (
            "You are a helpful assistant with access to various tools. "
            "Use the search_tools function to discover what tools are available "
            "before trying to use them. Search for tools based on the capability you need."
        )
        
        for iteration in range(self.max_iterations):
            print(f"\n[Advisor] === Iteration {iteration + 1} ===")
            print(f"[Advisor] Available tools: ['search_tools'] + {list(self.discovered_tools.keys())}")
            
            # Make API call
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt or default_system,
                tools=self._get_current_tools(),
                messages=messages
            )
            
            # Track tokens
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.request_count += 1
            
            print(f"[Advisor] Tokens this call: in={response.usage.input_tokens}, out={response.usage.output_tokens}")
            
            # Check if we're done (no tool use)
            if response.stop_reason == "end_turn":
                # Extract text response
                for block in response.content:
                    if block.type == "text":
                        print(f"\n[Advisor] === COMPLETE ===")
                        print(f"[Advisor] Total: {self.request_count} requests, "
                              f"{self.total_input_tokens} input tokens, "
                              f"{self.total_output_tokens} output tokens")
                        return block.text
                return ""
            
            # Process tool uses
            tool_results = []
            assistant_content = []
            
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
                    
                    # Execute the tool
                    result = self._handle_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add assistant message and tool results to conversation
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})
        
        return "Max iterations reached"


def create_example_tools() -> list[Tool]:
    """Create a sample set of tools for demonstration."""
    
    tools = [
        Tool(
            name="get_current_weather",
            description="Get the current weather conditions for a specific location",
            parameters={
                "properties": {
                    "location": {"type": "string", "description": "City name or coordinates"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units"}
                },
                "required": ["location"]
            },
            function=lambda location, units="celsius": {
                "location": location,
                "temperature": 22 if units == "celsius" else 72,
                "units": units,
                "conditions": "Sunny with light clouds",
                "humidity": 45
            }
        ),
        Tool(
            name="get_weather_forecast",
            description="Get weather forecast for the next 5 days for a location",
            parameters={
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Number of days (1-5)"}
                },
                "required": ["location"]
            },
            function=lambda location, days=5: {
                "location": location,
                "forecast": [{"day": i+1, "temp": 20+i, "conditions": "Partly cloudy"} for i in range(days)]
            }
        ),
        
        # time tools
        Tool(
            name="get_current_time",
            description="Get the current date and time for a specific timezone or location",
            parameters={
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone like 'Europe/Amsterdam' or 'America/New_York'"}
                },
                "required": ["timezone"]
            },
            function=lambda timezone: {
                "timezone": timezone,
                "datetime": "2025-01-05T14:30:00",
                "day_of_week": "Sunday"
            }
        ),
        
        # shopping tools
        Tool(
            name="find_clothing_stores",
            description="Find clothing and fashion stores in a specific location that are currently open",
            parameters={
                "properties": {
                    "location": {"type": "string", "description": "City or area"},
                    "open_now": {"type": "boolean", "description": "Only show stores currently open"}
                },
                "required": ["location"]
            },
            function=lambda location, open_now=True: {
                "location": location,
                "stores": [
                    {"name": "H&M", "open": True, "address": "123 Main St"},
                    {"name": "Zara", "open": True, "address": "456 Fashion Ave"},
                    {"name": "Uniqlo", "open": True, "address": "789 Style Blvd"}
                ]
            }
        ),
        
        # database tools
        Tool(
            name="query_database",
            description="Execute a read-only SQL query against the application database",
            parameters={
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT query"},
                    "limit": {"type": "integer", "description": "Maximum rows to return"}
                },
                "required": ["query"]
            },
            function=lambda query, limit=100: {"rows": [], "count": 0, "query": query}
        ),
        
        # email tools
        Tool(
            name="send_email",
            description="Send an email to one or more recipients",
            parameters={
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}, "description": "Recipient email addresses"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body content"}
                },
                "required": ["to", "subject", "body"]
            },
            function=lambda to, subject, body: {"status": "sent", "recipients": to}
        ),
        Tool(
            name="search_emails",
            description="Search through emails by sender, subject, or content",
            parameters={
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "folder": {"type": "string", "description": "Email folder to search"}
                },
                "required": ["query"]
            },
            function=lambda query, folder="inbox": {"results": [], "total": 0}
        ),
        
        # calendar tools
        Tool(
            name="get_calendar_events",
            description="Get calendar events for a specific date range",
            parameters={
                "properties": {
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["start_date", "end_date"]
            },
            function=lambda start_date, end_date: {"events": [], "count": 0}
        ),
        Tool(
            name="create_calendar_event",
            description="Create a new calendar event",
            parameters={
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "start": {"type": "string", "description": "Start datetime"},
                    "end": {"type": "string", "description": "End datetime"},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "Attendee emails"}
                },
                "required": ["title", "start", "end"]
            },
            function=lambda title, start, end, attendees=None: {"id": "evt_123", "status": "created"}
        ),
        
        # file tools
        Tool(
            name="read_file",
            description="Read contents of a file from the filesystem",
            parameters={
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            },
            function=lambda path: {"content": f"Contents of {path}", "size": 1024}
        ),
        Tool(
            name="write_file",
            description="Write content to a file",
            parameters={
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            },
            function=lambda path, content: {"status": "written", "bytes": len(content)}
        ),
        Tool(
            name="list_directory",
            description="List files and folders in a directory",
            parameters={
                "properties": {
                    "path": {"type": "string", "description": "Directory path"}
                },
                "required": ["path"]
            },
            function=lambda path: {"files": ["file1.txt", "file2.py"], "dirs": ["subdir"]}
        ),
        
        # math/calculation tools
        Tool(
            name="calculate",
            description="Perform mathematical calculations",
            parameters={
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                },
                "required": ["expression"]
            },
            function=lambda expression: {"result": eval(expression), "expression": expression}
        ),
        Tool(
            name="unit_converter",
            description="Convert between different units of measurement",
            parameters={
                "properties": {
                    "value": {"type": "number", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"}
                },
                "required": ["value", "from_unit", "to_unit"]
            },
            function=lambda value, from_unit, to_unit: {"result": value * 1.0, "from": from_unit, "to": to_unit}
        ),
        
        # Web/API tools
        Tool(
            name="fetch_url",
            description="Fetch content from a URL",
            parameters={
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            },
            function=lambda url: {"status": 200, "content": f"Content from {url}"}
        ),
        Tool(
            name="search_web",
            description="Search the web for information",
            parameters={
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            function=lambda query: {"results": [{"title": f"Result for {query}", "url": "https://example.com"}]}
        ),
        
        # Task/todo tools
        Tool(
            name="create_task",
            description="Create a new task or todo item",
            parameters={
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "due_date": {"type": "string", "description": "Due date"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["title"]
            },
            function=lambda title, due_date=None, priority="medium": {"id": "task_456", "status": "created"}
        ),
        Tool(
            name="list_tasks",
            description="List all tasks, optionally filtered by status",
            parameters={
                "properties": {
                    "status": {"type": "string", "enum": ["pending", "completed", "all"]}
                },
                "required": []
            },
            function=lambda status="all": {"tasks": [], "count": 0}
        ),
        
        # Communication tools
        Tool(
            name="send_slack_message",
            description="Send a message to a Slack channel or user",
            parameters={
                "properties": {
                    "channel": {"type": "string", "description": "Channel name or user ID"},
                    "message": {"type": "string", "description": "Message content"}
                },
                "required": ["channel", "message"]
            },
            function=lambda channel, message: {"status": "sent", "channel": channel}
        ),
        Tool(
            name="create_jira_ticket",
            description="Create a new Jira ticket",
            parameters={
                "properties": {
                    "project": {"type": "string", "description": "Project key"},
                    "summary": {"type": "string", "description": "Ticket summary"},
                    "description": {"type": "string", "description": "Detailed description"},
                    "type": {"type": "string", "enum": ["bug", "task", "story"]}
                },
                "required": ["project", "summary"]
            },
            function=lambda project, summary, description="", type="task": {"key": f"{project}-123"}
        ),
    ]
    
    return tools


def main():
    """Example demonstrating the Tool Search Tool pattern."""
    
    # Initialize Anthropic client
    client = Anthropic()

    client = ChatOpenAI(model="gpt-4o")
    
    
    # Create tool searcher (use keyword-based for no extra dependencies)
    # searcher = KeywordToolSearcher()
    
    # Or use vector search for better semantic matching:
    searcher = VectorToolSearcher()
    
    # Create the advisor
    advisor = ToolSearchToolAdvisor(
        client=client,
        tool_searcher=searcher,
        model="claude-sonnet-4-20250514",
        max_search_results=5
    )
    
    tools = create_example_tools()
    advisor.register_tools(tools)
    
    query = "Help me plan what to wear today in Amsterdam."
    print("\n" + "="*60)
    print(f"USER:{query}")
    print("Please suggest clothing shops that are open right now.")
    print("="*60)
    
    response = advisor.chat(
        "Help me plan what to wear today in Amsterdam. "
        "Please suggest clothing shops that are open right now."
    )
    
    print("\n" + "="*60)
    print("ASSISTANT:")
    print("="*60)
    print(response)


if __name__ == "__main__":
    main()