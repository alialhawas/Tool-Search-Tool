# Tool Search Tool Pattern

A Python implementation of dynamic tool discovery for LLM agents, based on [Anthropic's advanced tool use pattern](https://www.anthropic.com/engineering/advanced-tool-use).

## Overview

When working with large tool libraries (20+ tools), sending all tool schemas to the LLM wastes tokens and can degrade performance. This implementation solves that by:

1. Sending only a `search_tools` meta-tool initially
2. Letting the LLM search for relevant tools on demand
3. Dynamically injecting discovered tools into subsequent API calls

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│   ToolSearchToolAdvisor     │
│   (Orchestrator)            │
├─────────────────────────────┤
│ • Manages conversation loop │
│ • Tracks discovered tools   │
│ • Executes tool calls       │
│ • Tracks token usage        │
└──────────────┬──────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌────────────┐    ┌──────────────┐
│ search_    │    │ Discovered   │
│ tools      │    │ Tools        │
│ (meta-tool)│    │ (dynamic)    │
└─────┬──────┘    └──────────────┘
      │
      ▼
┌─────────────────────────────┐
│      ToolSearcher           │
│      (Abstract)             │
├─────────────────────────────┤
│ KeywordToolSearcher         │
│   └─ TF-IDF-like scoring    │
│ VectorToolSearcher          │
│   └─ Semantic embeddings    │
└─────────────────────────────┘
```

## Components

### `Tool` (dataclass)

Represents a callable tool with:
- `name`: Tool identifier
- `description`: What the tool does
- `parameters`: JSON Schema for input validation
- `function`: The actual callable

```python
Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={"properties": {"location": {"type": "string"}}, "required": ["location"]},
    function=lambda location: {"temp": 22, "location": location}
)
```

### `ToolSearcher` (abstract base class)

Interface for tool search strategies with two methods:
- `index(tools)`: Index all available tools
- `search(query, max_results)`: Find tools matching a query

### `KeywordToolSearcher`

Simple keyword-based search with no external dependencies:
- Tokenizes tool names and descriptions
- Scores by keyword overlap (normalized by query length)
- Fast but limited to exact/partial word matches

### `VectorToolSearcher`

Semantic search using sentence embeddings:
- Requires `numpy` and `sentence-transformers`
- Uses cosine similarity between query and tool embeddings
- Better at understanding intent (e.g., "send message" finds "send_email")

### `ToolSearchToolAdvisor`

The main orchestrator that:

1. **Registers tools** - Stores all tools and indexes them with the searcher
2. **Manages the conversation loop** - Iterates until the LLM produces a final response
3. **Handles tool calls**:
   - `search_tools`: Searches for tools and adds them to the discovered set
   - Other tools: Executes the tool function and returns results
4. **Tracks metrics** - Input/output tokens and request count

## Flow Example

```
User: "What's the weather in Amsterdam?"

Iteration 1:
  Available: [search_tools]
  LLM calls: search_tools(query="weather")
  Result: Discovers get_current_weather, get_weather_forecast

Iteration 2:
  Available: [search_tools, get_current_weather, get_weather_forecast]
  LLM calls: get_current_weather(location="Amsterdam")
  Result: {temperature: 22, conditions: "Sunny"}

Iteration 3:
  LLM returns final text response to user
```

## Usage

```python
from anthropic import Anthropic
from main import Tool, KeywordToolSearcher, ToolSearchToolAdvisor

# Initialize
client = Anthropic()
searcher = KeywordToolSearcher()
advisor = ToolSearchToolAdvisor(client=client, tool_searcher=searcher)

# Register tools
tools = [
    Tool(name="get_weather", description="Get weather", parameters={...}, function=...),
    # ... more tools
]
advisor.register_tools(tools)

# Chat
response = advisor.chat("What's the weather in Paris?")
print(response)
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `claude-sonnet-4-20250514` | Claude model to use |
| `max_search_results` | `5` | Max tools returned per search |
| `max_iterations` | `10` | Max conversation loop iterations |

## Dependencies

**Required:**
- `anthropic` - Anthropic API client

**Optional (for vector search):**
- `numpy`
- `sentence-transformers`

```bash
# Minimal
pip install anthropic

# With vector search
pip install anthropic numpy sentence-transformers
```

## Token Savings

With 20+ tools, sending all schemas might use 2000+ tokens per request. With this pattern:
- Initial request: ~100 tokens (just `search_tools`)
- After discovery: Only relevant tools are included
- Typical savings: 50-80% reduction in tool schema tokens

## Example Tools Included

The `create_example_tools()` function provides 20 sample tools across categories:
- Weather (current, forecast)
- Time/timezone
- Shopping (store finder)
- Email (send, search)
- Calendar (events, create)
- Files (read, write, list)
- Math (calculate, convert)
- Web (fetch, search)
- Tasks (create, list)
- Communication (Slack, Jira)
