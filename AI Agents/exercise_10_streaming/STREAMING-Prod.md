# Exercise 10: Streaming & Async Execution

## 🎯 Learning Objectives

By the end of this exercise, you will:

- ✅ Understand LangGraph's 5 streaming modes (`values`, `updates`, `messages`, `custom`, `debug`)
- ✅ Stream LLM responses token-by-token for responsive UIs
- ✅ Use `astream()` for async execution
- ✅ Implement custom progress updates during long operations
- ✅ Handle concurrent operations with `asyncio`
- ✅ Build production-ready streaming applications

---

## 📚 Theory: Why Streaming Matters

### The Problem with Blocking

```python
# ❌ Bad: User waits 10+ seconds for full response
result = agent.invoke({"query": "Explain quantum computing"})
print(result["answer"])  # Nothing shown until complete!
```

**Issues:**

- Poor UX (user sees nothing for seconds)
- Can't show progress
- Feels unresponsive
- Can't cancel mid-execution

### The Solution: Streaming

```python
# ✅ Good: User sees tokens as they're generated
for chunk in agent.stream({"query": "Explain quantum computing"}, stream_mode="messages"):
    print(chunk[0].content, end="", flush=True)  # Real-time output!
```

**Benefits:**

- ✅ Immediate feedback
- ✅ Progress indicators
- ✅ Responsive feel
- ✅ Can cancel early

---

## 🔧 LangGraph Streaming Modes

### 1. **`values` Mode** - Full State Snapshots

**Use Case:** Need complete state after each step

```python
for chunk in graph.stream(inputs, stream_mode="values"):
    print(chunk)  # Full state after each node
```

**Output:**

```python
{'topic': 'cats', 'joke': ''}
{'topic': 'cats and dogs', 'joke': ''}
{'topic': 'cats and dogs', 'joke': 'Why did the cat...'}
```

**When to use:**

- Building UIs that show full context
- Debugging state transitions
- Need to display all state variables

---

### 2. **`updates` Mode** - State Deltas

**Use Case:** Track what changed after each node

```python
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)  # Only changes
```

**Output:**

```python
{'refine_topic': {'topic': 'cats and dogs'}}
{'generate_joke': {'joke': 'Why did the cat...'}}
```

**When to use:**

- Efficient UI updates (only re-render what changed)
- Progress tracking
- Logging state changes

---

### 3. **`messages` Mode** - LLM Token Streaming ⭐

**Use Case:** Stream LLM output token-by-token

```python
for message_chunk, metadata in graph.stream(inputs, stream_mode="messages"):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)
```

**Output:**

```
Why| did| the| cat| sit| on| the| computer|?| To| keep| an| eye| on| the| mouse|!|
```

**When to use:**

- Chat interfaces
- Real-time text generation
- Responsive UIs

**Important:** Works even with `.invoke()` - LangGraph automatically streams tokens!

---

### 4. **`custom` Mode** - Progress Updates

**Use Case:** Emit custom progress/status during execution

```python
from langgraph.config import get_stream_writer

def long_operation(state):
    writer = get_stream_writer()

    writer({"status": "Starting..."})
    # Do work
    writer({"status": "50% complete"})
    # More work
    writer({"status": "Done!"})

    return {"result": "..."}

for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk)  # {'status': 'Starting...'}
```

**When to use:**

- Long-running operations
- Progress bars
- Status indicators
- Multi-step workflows

---

### 5. **`debug` Mode** - Full Execution Trace

**Use Case:** Deep debugging and observability

```python
for chunk in graph.stream(inputs, stream_mode="debug"):
    print(chunk)  # Node entry/exit, state, errors
```

**Output:**

```python
{'type': 'task', 'timestamp': '...', 'step': 1, 'payload': {...}}
{'type': 'task_result', 'timestamp': '...', 'step': 1, 'payload': {...}}
```

**When to use:**

- Debugging complex graphs
- Performance profiling
- Audit trails
- Understanding execution flow

---

## 🚀 Async Execution with `astream()`

### Why Async?

**Scenario:** Your agent needs to:

1. Search 3 different databases
2. Call 2 external APIs
3. Generate a response

**Synchronous (Slow):**

```python
# Total time: 15 seconds (5s + 5s + 5s)
result1 = search_db1()  # 5s
result2 = search_db2()  # 5s
result3 = search_db3()  # 5s
```

**Asynchronous (Fast):**

```python
# Total time: 5 seconds (all run concurrently!)
results = await asyncio.gather(
    search_db1(),
    search_db2(),
    search_db3()
)
```

### Using `astream()`

```python
import asyncio

async def main():
    async for chunk in graph.astream(inputs, stream_mode="messages"):
        print(chunk[0].content, end="", flush=True)

asyncio.run(main())
```

---

## 🎓 Key Concepts

### 1. **Combining Multiple Modes**

```python
for mode, chunk in graph.stream(inputs, stream_mode=["updates", "custom"]):
    if mode == "updates":
        print(f"State changed: {chunk}")
    elif mode == "custom":
        print(f"Progress: {chunk}")
```

### 2. **Filtering LLM Streams by Tags**

```python
model_1 = ChatOpenAI(model="gpt-4o", tags=["joke"])
model_2 = ChatOpenAI(model="gpt-4o", tags=["poem"])

async for msg, metadata in graph.astream(inputs, stream_mode="messages"):
    if metadata["tags"] == ["joke"]:
        print(msg.content, end="", flush=True)
```

### 3. **Custom Progress in Tools**

```python
from langchain.tools import tool
from langgraph.config import get_stream_writer

@tool
def process_data(data: str):
    """Process large dataset."""
    writer = get_stream_writer()

    for i in range(100):
        # Process chunk
        writer({"progress": f"{i+1}/100", "type": "progress"})

    return "Done"
```

---

## 🏗️ Progressive Tasks

### **Task 1: Basic Token Streaming** ⭐

**Goal:** Convert your RAG agent to stream responses token-by-token

**Subtasks:**

- [ ] Use `stream_mode="messages"` for LLM output
- [ ] Display tokens in real-time with `flush=True`
- [ ] Test with long responses
- [ ] Add typing indicator before first token

**Success Criteria:**

- ✅ Tokens appear progressively (not all at once)
- ✅ No buffering delay
- ✅ Works with multi-turn conversations

---

### **Task 2: Async Retrieval** ⭐⭐

**Goal:** Make retrieval from multiple collections concurrent

**Subtasks:**

- [ ] Convert retrieval functions to async
- [ ] Use `asyncio.gather()` for parallel retrieval
- [ ] Measure time savings (sequential vs parallel)
- [ ] Handle errors in concurrent operations

**Success Criteria:**

- ✅ Multiple retrievals happen simultaneously
- ✅ Total time < sum of individual times
- ✅ Errors in one retrieval don't crash others

---

### **Task 3: Custom Progress Updates** ⭐⭐

**Goal:** Show "Routing...", "Retrieving...", "Generating..." status

**Subtasks:**

- [ ] Use `get_stream_writer()` in each node
- [ ] Emit progress updates with `stream_mode="custom"`
- [ ] Build CLI progress display
- [ ] Add timestamps to progress updates

**Success Criteria:**

- ✅ User sees what the agent is doing
- ✅ Progress updates appear in correct order
- ✅ Final answer still streams token-by-token

---

### **Task 4: Multi-Mode Streaming** ⭐⭐⭐

**Goal:** Combine `updates`, `messages`, and `custom` modes

**Subtasks:**

- [ ] Stream with `["updates", "messages", "custom"]`
- [ ] Build UI that shows:
  - State changes (updates)
  - LLM tokens (messages)
  - Progress (custom)
- [ ] Handle mode filtering correctly
- [ ] Add colored output for different modes

**Success Criteria:**

- ✅ All three modes work simultaneously
- ✅ UI clearly distinguishes between modes
- ✅ No performance degradation

---

### **Task 5: Production Features** ⭐⭐⭐

**Goal:** Add rate limiting, error handling, cancellation

**Subtasks:**

- [ ] Implement rate limiting (max 10 req/min)
- [ ] Add timeout handling (cancel after 30s)
- [ ] Handle stream interruptions gracefully
- [ ] Add retry logic for failed streams
- [ ] Log streaming metrics (tokens/sec, latency)

**Success Criteria:**

- ✅ Rate limiting prevents API overload
- ✅ Timeouts work correctly
- ✅ Stream errors don't crash app
- ✅ Metrics logged for monitoring

---

## ⚠️ Common Pitfalls

### 1. **Forgetting `flush=True`**

```python
# ❌ Buffered - tokens don't appear immediately
print(chunk.content, end="")

# ✅ Flushed - tokens appear instantly
print(chunk.content, end="", flush=True)
```

### 2. **Not Handling Empty Chunks**

```python
# ❌ Crashes on empty content
print(chunk.content, end="", flush=True)

# ✅ Check for content first
if chunk.content:
    print(chunk.content, end="", flush=True)
```

### 3. **Mixing Sync and Async**

```python
# ❌ Can't use await in sync function
def my_function():
    result = await graph.astream(...)  # Error!

# ✅ Use async function
async def my_function():
    async for chunk in graph.astream(...):
        print(chunk)
```

### 4. **Python Version Issues**

**For Python < 3.11:** Must pass `config` explicitly for streaming

```python
# Python 3.11+: Works automatically
async for chunk in graph.astream(inputs, stream_mode="messages"):
    ...

# Python < 3.11: Need explicit config
config = {"configurable": {"thread_id": "1"}}
async for chunk in graph.astream(inputs, config=config, stream_mode="messages"):
    ...
```

---

## 🧠 Knowledge Check

Before starting, answer these:

1. **Q:** What's the difference between `values` and `updates` mode?
   **A:** `values` streams full state, `updates` streams only changes

2. **Q:** Can you stream LLM tokens even if you use `.invoke()` instead of `.stream()`?
   **A:** Yes! LangGraph automatically streams tokens in `messages` mode

3. **Q:** How do you emit custom progress updates?
   **A:** Use `get_stream_writer()` and call `writer({"key": "value"})`

4. **Q:** Why use async execution?
   **A:** To run independent operations concurrently (faster total time)

5. **Q:** What's the benefit of combining multiple stream modes?
   **A:** Get different types of updates (state, tokens, progress) in one stream

---

## 📖 Official Documentation

- [LangGraph Streaming](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Stream Modes](https://docs.langchain.com/oss/python/langgraph/streaming#supported-stream-modes)
- [Async Execution](https://docs.langchain.com/oss/python/langgraph/streaming#async-with-python--3-11)

---

## 🚀 Next Steps

1. Read this README thoroughly
2. Understand each streaming mode
3. Start with Task 1 (Token Streaming)
4. Progress through tasks sequentially
5. Test with real-world scenarios

Let's build responsive, production-grade agents! 💪
