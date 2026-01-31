# Streaming Quick Reference

## 🎯 When to Use Each Mode

| Mode       | Use Case             | Output Type               | Best For                           |
| ---------- | -------------------- | ------------------------- | ---------------------------------- |
| `values`   | Full state snapshots | Complete state dict       | Debugging, full context UIs        |
| `updates`  | State changes only   | Delta dict                | Efficient UIs, progress tracking   |
| `messages` | LLM token streaming  | (message_chunk, metadata) | Chat interfaces, responsive UIs    |
| `custom`   | Progress updates     | Custom dict               | Status indicators, long operations |
| `debug`    | Execution trace      | Debug events              | Debugging, observability           |

---

## 📝 Code Snippets

### Basic Token Streaming

```python
for msg, metadata in graph.stream(inputs, stream_mode="messages"):
    if msg.content:
        print(msg.content, end="", flush=True)
```

### Custom Progress

```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()
    writer({"status": "Processing..."})
    # do work
    return {"result": "..."}

for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk["status"])
```

### Multi-Mode

```python
for mode, chunk in graph.stream(inputs, stream_mode=["updates", "messages", "custom"]):
    if mode == "updates":
        print(f"State: {chunk}")
    elif mode == "messages":
        print(chunk[0].content, end="", flush=True)
    elif mode == "custom":
        print(f"Progress: {chunk}")
```

### Async Streaming

```python
async def main():
    async for msg, metadata in graph.astream(inputs, stream_mode="messages"):
        if msg.content:
            print(msg.content, end="", flush=True)

asyncio.run(main())
```

### Concurrent Operations

```python
async def parallel_retrieval():
    results = await asyncio.gather(
        retrieve_from_db1(),
        retrieve_from_db2(),
        retrieve_from_db3(),
        return_exceptions=True
    )
    return results
```

---

## ⚠️ Common Mistakes

### 1. Forgetting flush

```python
# ❌ Buffered
print(chunk, end="")

# ✅ Immediate
print(chunk, end="", flush=True)
```

### 2. Not checking for empty chunks

```python
# ❌ Can crash
print(chunk.content, end="", flush=True)

# ✅ Safe
if chunk.content:
    print(chunk.content, end="", flush=True)
```

### 3. Mixing sync/async

```python
# ❌ Error
def my_func():
    await graph.astream(...)

# ✅ Correct
async def my_func():
    async for chunk in graph.astream(...):
        ...
```

---

## 🎯 Performance Tips

1. **Use `messages` mode for UX** - Users see immediate feedback
2. **Use `updates` mode for efficiency** - Only send changes, not full state
3. **Combine modes sparingly** - Each mode adds overhead
4. **Use async for I/O** - Concurrent operations are much faster
5. **Add rate limiting** - Prevent API overload

---

## 📊 Metrics to Track

```python
class StreamMetrics:
    start_time: float          # When stream started
    first_token_time: float    # When first token arrived
    token_count: int           # Total tokens streamed
    error_count: int           # Number of errors

    @property
    def ttft(self):
        """Time to first token"""
        return self.first_token_time - self.start_time

    @property
    def tokens_per_second(self):
        elapsed = time.time() - self.start_time
        return self.token_count / elapsed
```

---

## 🧪 Testing Checklist

- [ ] Short queries (< 10 tokens)
- [ ] Long queries (> 100 tokens)
- [ ] Multi-turn conversations
- [ ] Empty responses
- [ ] Error scenarios
- [ ] Timeout scenarios
- [ ] Rate limit scenarios
- [ ] Concurrent requests

---

## 📖 Official Docs

- [Streaming Guide](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Stream Modes](https://docs.langchain.com/oss/python/langgraph/streaming#supported-stream-modes)
- [Async Execution](https://docs.langchain.com/oss/python/langgraph/streaming#async-with-python--3-11)
