# Exercise 9B: Interrupts in Tools

## 🎯 Learning Objectives

By the end of this exercise, you will:

- ✅ Understand how to use `interrupt()` inside tool functions
- ✅ Implement tool call approval workflows
- ✅ Allow humans to edit tool parameters before execution
- ✅ Handle tool call cancellation gracefully
- ✅ Build production-ready tool approval systems

---

## 📚 Theory: Interrupts in Tools

### What Are Tool Interrupts?

In Exercise 9, you learned to use `interrupt()` in **nodes**. But what if you want to pause execution **inside a tool call**?

**Use Cases:**

- 🔒 Approve sensitive API calls (payments, deletions)
- ✉️ Review emails before sending
- 📝 Edit tool parameters before execution
- 🛡️ Add human oversight to risky operations

### Key Difference: Nodes vs Tools

| Aspect       | Interrupts in Nodes | Interrupts in Tools      |
| ------------ | ------------------- | ------------------------ |
| **Location** | Inside graph nodes  | Inside `@tool` functions |
| **When**     | Between steps       | During tool execution    |
| **Use Case** | Workflow approval   | Action approval          |
| **Example**  | Approve SQL query   | Approve email sending    |

---

## 🔧 How It Works

### Basic Pattern

```python
from langchain.tools import tool
from langgraph.types import interrupt

@tool
def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""

    # Pause before sending
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this email?"
    })

    # Check approval
    if response.get("action") == "approve":
        # User can override parameters
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)

        # Actually send (your implementation)
        return f"Email sent to {final_to}"

    return "Email cancelled by user"
```

### Resume Pattern

```python
# Initial run - hits interrupt
config = {"configurable": {"thread_id": "email-1"}}
result = agent.invoke({"messages": [...]}, config=config)

# Check interrupt payload
print(result["__interrupt__"])
# → [Interrupt(value={'action': 'send_email', 'to': '...', ...})]

# Option 1: Approve as-is
agent.invoke(Command(resume={"action": "approve"}), config=config)

# Option 2: Approve with edits
agent.invoke(
    Command(resume={
        "action": "approve",
        "subject": "Updated Subject",  # Override
        "body": "Updated body"
    }),
    config=config
)

# Option 3: Cancel
agent.invoke(Command(resume={"action": "cancel"}), config=config)
```

---

## 🎓 Key Concepts

### 1. **Idempotency in Tools**

Just like nodes, code before `interrupt()` runs twice:

```python
@tool
def risky_operation(data: str):
    print("This prints TWICE!")  # ⚠️ Runs on initial + resume

    response = interrupt({"data": data})

    print("This prints ONCE")  # ✅ Only after resume
    return process(data)
```

**Best Practice:** Keep side effects AFTER `interrupt()`.

### 2. **Parameter Override Pattern**

Allow users to edit tool inputs:

```python
response = interrupt({"to": to, "subject": subject})

# Use .get() with fallback to original
final_to = response.get("to", to)  # User's edit OR original
final_subject = response.get("subject", subject)
```

### 3. **Graceful Cancellation**

Always provide a way to cancel:

```python
if response.get("action") == "approve":
    # Execute
    return execute_action()
else:
    # Cancel
    return "Action cancelled by user"
```

---

## 🏗️ Progressive Tasks

### **Task 1: Basic Tool Approval**

Build an email agent that:

- LLM decides to send an email
- Pauses for human approval
- Sends email if approved
- Cancels if rejected

**Files to create:**

- `task_1_email_approval.py`

---

### **Task 2: Parameter Editing**

Extend Task 1 to allow editing:

- User can modify recipient
- User can edit subject
- User can change body
- Execute with modified parameters

---

### **Task 3: Multiple Tools**

Build an agent with multiple tools:

- `send_email` - requires approval
- `search_web` - no approval needed
- `delete_record` - requires approval
- Conditional interrupts based on tool

---

### **Task 4: Production Hardening**

Add production features:

- Logging (who approved what, when)
- Timeout handling (auto-cancel after 60s)
- Validation (check email format, etc.)
- Error handling

---

## 🎯 Real-World Scenarios

### Scenario 1: Payment Processing

```python
@tool
def process_payment(amount: float, recipient: str):
    """Process a payment."""
    response = interrupt({
        "action": "payment",
        "amount": amount,
        "recipient": recipient,
        "message": f"Approve payment of ${amount} to {recipient}?"
    })

    if response.get("action") == "approve":
        # Process payment
        return f"Paid ${amount} to {recipient}"
    return "Payment cancelled"
```

### Scenario 2: Database Deletion

```python
@tool
def delete_user(user_id: int):
    """Delete a user from database."""
    response = interrupt({
        "action": "delete_user",
        "user_id": user_id,
        "message": f"⚠️ Permanently delete user {user_id}?"
    })

    if response.get("action") == "approve":
        # Delete from DB
        return f"User {user_id} deleted"
    return "Deletion cancelled"
```

---

## ⚠️ Common Pitfalls

### 1. **Forgetting Checkpointer**

```python
# ❌ Won't work - no persistence
graph = builder.compile()

# ✅ Correct
checkpointer = SqliteSaver(sqlite3.connect("tools.db"))
graph = builder.compile(checkpointer=checkpointer)
```

### 2. **Not Handling Cancellation**

```python
# ❌ Always executes
response = interrupt({"action": "send"})
send_email()  # Sends even if cancelled!

# ✅ Check approval first
if response.get("action") == "approve":
    send_email()
```

### 3. **Side Effects Before Interrupt**

```python
# ❌ Sends twice!
send_email()
response = interrupt({"message": "Email sent, approve?"})

# ✅ Interrupt first, then send
response = interrupt({"message": "Approve sending?"})
if response.get("action") == "approve":
    send_email()
```

---

## 🧠 Knowledge Check

Before starting, answer these:

1. **Q:** Where does `interrupt()` go in a tool function?
   **A:** Before the actual action (API call, DB write, etc.)

2. **Q:** How do you allow editing tool parameters?
   **A:** Use `response.get("param", original_value)` pattern

3. **Q:** What happens if you don't check for approval?
   **A:** Tool executes even if user cancelled!

4. **Q:** Why do you need a checkpointer for tool interrupts?
   **A:** To persist state during the pause

---

## 📖 Official Documentation

- [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Interrupts in Tools](https://docs.langchain.com/oss/python/langgraph/interrupts#interrupts-in-tools)

---

## 🚀 Next Steps

1. Read this README thoroughly
2. Start with Task 1 (Basic Tool Approval)
3. Test approve and reject flows
4. Move to Task 2 (Parameter Editing)
5. Build production features in Tasks 3-4

Let's build! 💪
