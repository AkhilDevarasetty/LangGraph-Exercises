# Exercise 9: Human-in-the-Loop (HITL) Patterns

**Objective:** Build production-grade agents that pause for human approval before executing critical actions, ensuring safety and oversight in autonomous systems.

**Priority:** 🔴 CRITICAL  
**Estimated Time:** 4-5 hours  
**Difficulty:** ⭐⭐⭐⭐

---

## 📚 Theory & Concepts

### **What is Human-in-the-Loop (HITL)?**

Human-in-the-Loop is a design pattern where AI agents pause execution at critical decision points to request human approval, input, or modification before proceeding. This is **absolutely essential** for production systems that take actions with real-world consequences.

**Think of it like this:**  
Imagine you're using a smart home assistant that can order groceries. You say "I need milk." The AI:

- **Without HITL:** Immediately orders 10 gallons of milk to your address. Oops!
- **With HITL:** Shows you the order (2 gallons, $8.99, delivery tomorrow) and asks "Approve this order?" You can approve, modify, or reject.

The same principle applies to AI agents in production. Any action that affects data, costs money, or impacts users should require human approval.

### **Why HITL Matters in Production**

In real-world applications, HITL enables five critical safety and control capabilities:

#### **1. Safety & Risk Mitigation**

Prevent catastrophic errors by requiring human approval before dangerous actions.

**Real-world scenario:**  
An AI agent managing a database generates this SQL query:

```sql
DELETE FROM users WHERE last_login < '2024-01-01';
```

Without HITL, it executes immediately and deletes 50,000 inactive users (including your CEO who was on sabbatical). With HITL, a human reviews the query, realizes the mistake, and modifies it to archive instead of delete.

#### **2. Compliance & Audit Trail**

Many industries (healthcare, finance, legal) require human oversight for regulatory compliance.

**Real-world scenario:**  
A medical AI agent suggests a treatment plan. Healthcare regulations require a licensed physician to review and approve the plan before implementation. HITL provides the approval checkpoint and creates an audit trail showing who approved what and when.

#### **3. Quality Control**

Ensure AI outputs meet quality standards before they reach end users.

**Real-world scenario:**  
An AI content generator creates a blog post for your company website. Before publishing, a human editor reviews it for:

- Factual accuracy
- Brand voice consistency
- Legal compliance
- SEO optimization

The editor can approve as-is, request revisions, or reject entirely.

#### **4. Learning & Feedback Loop**

Human decisions during HITL create training data to improve the AI over time.

**Real-world scenario:**  
An AI email assistant drafts responses to customer support tickets. When humans consistently modify certain types of responses (e.g., adding more empathy to refund requests), this feedback can be used to fine-tune the AI's behavior.

#### **5. Trust & Transparency**

Users trust AI systems more when they know humans are in the loop for critical decisions.

**Real-world scenario:**  
A loan approval AI makes decisions that affect people's lives. By requiring human review for edge cases (unusual income sources, first-time borrowers), the bank builds trust and can explain decisions to customers.

---

### **How HITL Works in LangGraph**

LangGraph implements HITL through the `interrupt()` function and `Command` primitive. Here's the execution flow:

```
User: "Transfer $500 to Alice"
    ↓
[Node: parse_request] → Checkpoint saved
    ↓
[Node: validate_transfer] → Checkpoint saved
    ↓
[Node: approval_node] → interrupt() called → PAUSED ⏸️
    │
    │ (Graph execution suspended, state saved)
    │ (Human reviews: "Transfer $500 to Alice (account #12345)")
    │
    ↓ (Human approves)
[Node: approval_node] → Resumes with approval
    ↓
[Node: execute_transfer] → Checkpoint saved
    ↓
Response: "Transfer complete!"
```

#### **The Interrupt Mechanism**

When `interrupt()` is called:

1. **Execution pauses** at the exact line where `interrupt()` is called
2. **State is saved** using the checkpointer (requires persistent checkpointer in production)
3. **Payload is returned** to the caller under `__interrupt__` key (can be any JSON-serializable value)
4. **Graph waits** indefinitely until you resume with a response
5. **Response is injected** back into the node when resumed, becoming the return value of `interrupt()`

**Code example:**

```python
from langgraph.types import interrupt

def approval_node(state: State):
    # Pause execution and ask for approval
    approved = interrupt({
        "question": "Do you approve this action?",
        "details": state["action_details"]
    })

    # This line doesn't execute until human responds
    # 'approved' will be whatever the human passes via Command(resume=...)
    if approved:
        return {"status": "approved"}
    else:
        return {"status": "rejected"}
```

#### **Resuming with Command**

To resume a paused graph, use the `Command` primitive:

```python
from langgraph.types import Command

# Initial run - hits the interrupt and pauses
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"input": "data"}, config=config)

# Check what was interrupted
print(result["__interrupt__"])
# → [Interrupt(value={'question': 'Do you approve...', 'details': ...})]

# Resume with human's decision
# The value passed to resume becomes the return value of interrupt()
graph.invoke(Command(resume=True), config=config)  # Approve
# OR
graph.invoke(Command(resume=False), config=config)  # Reject
```

**Critical rules for resuming:**

- ✅ **Must use the same thread_id** when resuming
- ✅ **Resume value can be any JSON-serializable data** (bool, string, dict, list)
- ✅ **Node restarts from the beginning** when resumed (code before `interrupt()` runs again)
- ✅ **Must have a checkpointer configured** (MemorySaver for dev, SqliteSaver/PostgresSaver for production)

---

### **HITL Patterns: When to Interrupt**

Not every action needs human approval. Here's when to use HITL:

#### **Pattern 1: Critical Actions (Always Interrupt)**

Actions that are irreversible, costly, or affect external systems:

- **Database mutations:** DELETE, UPDATE, DROP statements
- **Financial transactions:** Payments, transfers, refunds
- **External API calls:** Sending emails, posting to social media, creating tickets
- **File operations:** Deleting files, overwriting data
- **User-facing changes:** Publishing content, changing settings

**Example:**

```python
def execute_sql_node(state: State):
    query = state["generated_query"]

    # ALWAYS interrupt before executing SQL
    approved = interrupt({
        "type": "sql_approval",
        "query": query,
        "affected_tables": state["tables"],
        "estimated_rows": state["row_count"]
    })

    if approved:
        result = execute_query(query)
        return {"result": result}
    else:
        return {"result": "Query rejected by user"}
```

#### **Pattern 2: Confidence-Based Interrupts (Conditional)**

Interrupt only when the AI's confidence is below a threshold:

```python
def generate_response_node(state: State):
    response = llm.invoke(state["messages"])
    confidence = calculate_confidence(response)

    if confidence < 0.8:  # Low confidence
        # Ask human to review
        approved_response = interrupt({
            "type": "low_confidence_review",
            "response": response,
            "confidence": confidence
        })
        return {"response": approved_response}
    else:
        # High confidence - proceed automatically
        return {"response": response}
```

#### **Pattern 3: Policy-Based Interrupts**

Interrupt based on business rules or policies:

```python
def transfer_money_node(state: State):
    amount = state["amount"]

    # Policy: Transfers over $1000 require approval
    if amount > 1000:
        approved = interrupt({
            "type": "high_value_transfer",
            "amount": amount,
            "recipient": state["recipient"]
        })
        if not approved:
            return {"status": "rejected"}

    # Proceed with transfer
    execute_transfer(amount, state["recipient"])
    return {"status": "completed"}
```

#### **Pattern 4: Review and Edit**

Allow humans to modify AI outputs before proceeding:

```python
def draft_email_node(state: State):
    draft = llm.invoke(state["context"])

    # Let human review and edit the draft
    final_email = interrupt({
        "type": "email_review",
        "draft": draft,
        "editable": True  # Signal that user can modify
    })

    # final_email will be either the original draft or modified version
    send_email(final_email)
    return {"sent_email": final_email}
```

---

### **Advanced HITL Patterns**

#### **Multi-Step Approval Workflows**

Some actions require multiple approval stages:

```python
def multi_stage_approval(state: State):
    # Stage 1: Technical review
    tech_approved = interrupt({
        "stage": "technical_review",
        "details": state["technical_details"]
    })

    if not tech_approved:
        return Command(goto="rejected")

    # Stage 2: Business review
    business_approved = interrupt({
        "stage": "business_review",
        "details": state["business_impact"]
    })

    if not business_approved:
        return Command(goto="rejected")

    # Both approved - proceed
    return Command(goto="execute")
```

#### **Timeout Handling**

In production, you can't wait forever for human approval:

```python
import time
from datetime import datetime, timedelta

def approval_with_timeout_node(state: State):
    # Record when we started waiting
    state["approval_requested_at"] = datetime.now().isoformat()

    approved = interrupt({
        "type": "approval_request",
        "timeout_seconds": 3600,  # 1 hour
        "details": state["action_details"]
    })

    # When resumed, check if timeout exceeded
    requested_at = datetime.fromisoformat(state["approval_requested_at"])
    if datetime.now() - requested_at > timedelta(hours=1):
        # Timeout exceeded - auto-reject
        return {"status": "timeout_rejected"}

    return {"status": "approved" if approved else "rejected"}
```

**Note:** LangGraph doesn't have built-in timeout support. You need to implement this in your application layer (e.g., a background job that auto-rejects after timeout).

#### **Batch Approvals**

Approve multiple actions at once:

```python
def batch_approval_node(state: State):
    actions = state["pending_actions"]  # List of actions

    # Present all actions for review
    decisions = interrupt({
        "type": "batch_approval",
        "actions": actions,
        "count": len(actions)
    })

    # decisions should be a list: [True, False, True, ...]
    approved_actions = [
        action for action, approved in zip(actions, decisions)
        if approved
    ]

    return {"approved_actions": approved_actions}
```

---

### **Interrupt vs. Conditional Edges**

**When to use `interrupt()`:**

- ✅ Need human input/approval
- ✅ Decision requires external context (not in state)
- ✅ Want to pause and resume later (even after server restart)

**When to use conditional edges:**

- ✅ Decision is purely based on state
- ✅ No human input needed
- ✅ Deterministic routing logic

**Example comparison:**

```python
# ❌ DON'T use interrupt for simple routing
def router_node(state: State):
    # This should be a conditional edge, not an interrupt!
    route = interrupt("Which route should I take?")
    return Command(goto=route)

# ✅ DO use conditional edge
def router_node(state: State):
    if state["query_type"] == "sql":
        return Command(goto="sql_handler")
    else:
        return Command(goto="general_handler")

# ✅ DO use interrupt for human decisions
def approval_node(state: State):
    # Human needs to review the generated SQL
    approved = interrupt({
        "query": state["sql_query"],
        "risk_level": "high"
    })
    return Command(goto="execute" if approved else "reject")
```

---

### **Production Best Practices**

#### **✅ Best Practice 1: Always Use Persistent Checkpointers**

```python
# ❌ DON'T use MemorySaver in production
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()  # Lost on restart!

# ✅ DO use SqliteSaver or PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

**Why:** If your server restarts while waiting for approval, MemorySaver loses all state. The user's approval request disappears.

#### **✅ Best Practice 2: Include Rich Context in Interrupts**

```python
# ❌ DON'T provide minimal context
approved = interrupt("Approve?")

# ✅ DO provide rich, actionable context
approved = interrupt({
    "type": "sql_approval",
    "query": state["sql_query"],
    "affected_tables": ["users", "orders"],
    "estimated_rows": 1500,
    "risk_level": "medium",
    "requested_by": state["user_id"],
    "timestamp": datetime.now().isoformat()
})
```

**Why:** Humans need context to make informed decisions. Rich payloads enable better UI/UX.

#### **✅ Best Practice 3: Validate Resume Values**

```python
def approval_node(state: State):
    response = interrupt({"question": "Approve?"})

    # ✅ Validate the response
    if not isinstance(response, bool):
        raise ValueError(f"Expected bool, got {type(response)}")

    return {"approved": response}
```

**Why:** Humans might pass unexpected values. Validate to prevent downstream errors.

#### **✅ Best Practice 4: Log All Approval Decisions**

```python
import logging

def approval_node(state: State):
    approved = interrupt({
        "action": state["action"],
        "details": state["details"]
    })

    # ✅ Log the decision for audit trail
    logging.info(
        f"Approval decision: {approved} for action {state['action']} "
        f"by user {state['user_id']} at {datetime.now()}"
    )

    return {"approved": approved}
```

**Why:** Audit trails are critical for compliance, debugging, and accountability.

#### **✅ Best Practice 5: Handle Idempotency**

```python
def approval_node(state: State):
    # ⚠️ This code runs TWICE: once before interrupt, once after resume

    # ❌ DON'T do side effects before interrupt
    send_notification("Approval requested")  # Sent twice!

    approved = interrupt({"question": "Approve?"})

    # ✅ DO side effects after interrupt
    if approved:
        send_notification("Action approved")  # Sent once

    return {"approved": approved}
```

**Why:** When a node resumes, it restarts from the beginning. Code before `interrupt()` runs again.

**Solution:** Use state to track if side effects already happened:

```python
def approval_node(state: State):
    # Only send notification if not already sent
    if not state.get("notification_sent"):
        send_notification("Approval requested")
        state["notification_sent"] = True

    approved = interrupt({"question": "Approve?"})

    return {"approved": approved}
```

#### **✅ Best Practice 6: Implement Approval Timeouts**

```python
# In your application layer (not in the graph)
import asyncio

async def wait_for_approval_with_timeout(thread_id, timeout_seconds=3600):
    """Wait for approval or timeout after 1 hour."""
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        # Check if human has responded
        state = checkpointer.get(thread_id)
        if state.get("approval_received"):
            return state["approval_decision"]

        await asyncio.sleep(10)  # Check every 10 seconds

    # Timeout - auto-reject
    graph.invoke(Command(resume=False), config={"configurable": {"thread_id": thread_id}})
    return False
```

---

### **Common Pitfalls & How to Avoid Them**

#### **❌ Pitfall 1: Forgetting to Configure Checkpointer**

```python
# ❌ This will crash
graph = builder.compile()  # No checkpointer!
result = graph.invoke({"input": "data"}, config=config)
# → Error: interrupt() requires a checkpointer
```

**Fix:**

```python
# ✅ Always configure checkpointer
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)
```

#### **❌ Pitfall 2: Using Different Thread IDs**

```python
# ❌ Different thread IDs
config1 = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"input": "data"}, config=config1)

config2 = {"configurable": {"thread_id": "thread-2"}}  # Different!
graph.invoke(Command(resume=True), config=config2)  # Won't work!
```

**Fix:**

```python
# ✅ Same thread ID for initial run and resume
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"input": "data"}, config=config)
graph.invoke(Command(resume=True), config=config)  # Same config
```

#### **❌ Pitfall 3: Non-Serializable Interrupt Payloads**

```python
# ❌ Can't serialize database connection
approved = interrupt({
    "db_connection": db_conn,  # Not JSON-serializable!
    "query": "SELECT * FROM users"
})
```

**Fix:**

```python
# ✅ Only JSON-serializable data
approved = interrupt({
    "query": "SELECT * FROM users",
    "database": "production",
    "table": "users"
})
```

#### **❌ Pitfall 4: Interrupting Inside Try/Except**

```python
# ❌ DON'T wrap interrupt in try/except
def approval_node(state: State):
    try:
        approved = interrupt("Approve?")
    except Exception as e:
        return {"error": str(e)}
```

**Why:** LangGraph uses exceptions internally to implement interrupts. Catching them breaks the mechanism.

**Fix:**

```python
# ✅ Let interrupt exceptions propagate
def approval_node(state: State):
    approved = interrupt("Approve?")
    return {"approved": approved}
```

---

## 🧠 Knowledge Check Questions

Before we start coding, think through these scenarios:

### **Question 1: When to Use HITL**

You're building an AI customer support agent. Which of these actions should require human approval?

A) Answering "What are your business hours?"  
B) Issuing a $50 refund to a customer  
C) Searching the knowledge base for troubleshooting steps  
D) Updating the customer's email address in the database

<details>
<summary>Click for answer</summary>

**B and D require human approval.**

- **A (No HITL):** Simple factual question, no risk
- **B (HITL):** Financial transaction, requires approval
- **C (No HITL):** Read-only operation, no risk
- **D (HITL):** Data mutation, potential for error (wrong email)

**General rule:** Interrupt for actions that are:

- Irreversible (refunds, deletions)
- Costly (financial transactions)
- Affect external systems (database writes, API calls)

</details>

---

### **Question 2: Interrupt Payload Design**

Which interrupt payload is better for a SQL approval workflow?

**Option A:**

```python
approved = interrupt("Approve this query?")
```

**Option B:**

```python
approved = interrupt({
    "type": "sql_approval",
    "query": "DELETE FROM users WHERE inactive = true",
    "affected_rows": 1500,
    "tables": ["users"],
    "risk": "high"
})
```

<details>
<summary>Click for answer</summary>

**Option B is much better.**

**Why:**

- Provides rich context for the human to make an informed decision
- Enables better UI (can show query, risk level, affected rows)
- Creates better audit trail (logs contain full context)
- Allows for conditional UI (high risk → show warning)

**Option A problems:**

- No context for decision
- Human doesn't know what they're approving
- Poor audit trail
- Can't build good UI around it

</details>

---

### **Question 3: Idempotency Issue**

What's wrong with this code?

```python
def approval_node(state: State):
    # Send email notification
    send_email(state["user_email"], "Approval needed")

    approved = interrupt({"question": "Approve?"})

    if approved:
        execute_action(state["action"])

    return {"approved": approved}
```

<details>
<summary>Click for answer</summary>

**Problem:** The email is sent **twice**.

**Why:** When the node resumes after `interrupt()`, it restarts from the beginning. The `send_email()` call runs again.

**Fix:**

```python
def approval_node(state: State):
    # Only send email if not already sent
    if not state.get("email_sent"):
        send_email(state["user_email"], "Approval needed")
        state["email_sent"] = True

    approved = interrupt({"question": "Approve?"})

    if approved:
        execute_action(state["action"])

    return {"approved": approved}
```

**Key lesson:** Code before `interrupt()` must be idempotent (safe to run multiple times).

</details>

---

### **Question 4: Thread ID Mistake**

A user requests approval at 2pm. You restart your server at 3pm. At 4pm, the user approves. What happens?

**Assumptions:**

- Using `SqliteSaver` checkpointer
- Same `thread_id` used for initial run and resume
- Server restart cleared all in-memory state

<details>
<summary>Click for answer</summary>

**It works perfectly!** The approval succeeds and the graph resumes.

**Why:**

- `SqliteSaver` persists state to disk (survives restarts)
- Thread ID links the resume to the original execution
- Checkpoint contains all state needed to resume

**If you used `MemorySaver`:**

- ❌ State lost on restart
- ❌ Resume fails (no checkpoint found)
- ❌ User's approval is lost

**This is why persistent checkpointers are critical for HITL in production.**

</details>

---

### **Question 5: Multiple Interrupts**

Can a single node have multiple `interrupt()` calls? What happens?

```python
def multi_approval_node(state: State):
    tech_approved = interrupt({"stage": "technical"})

    if not tech_approved:
        return {"status": "rejected"}

    business_approved = interrupt({"stage": "business"})

    return {"status": "approved" if business_approved else "rejected"}
```

<details>
<summary>Click for answer</summary>

**Yes, this works!** The node pauses at each `interrupt()` sequentially.

**Execution flow:**

1. First run: Pauses at `interrupt({"stage": "technical"})`
2. Resume with `True`: Continues to `interrupt({"stage": "business"})`
3. Resume with `True`: Completes and returns `{"status": "approved"}`

**Important notes:**

- Each `interrupt()` requires a separate `resume`
- Node restarts from the beginning each time
- State must track which stage you're at to avoid re-running earlier interrupts

**Better pattern:**

```python
def multi_approval_node(state: State):
    if not state.get("tech_approved"):
        tech_approved = interrupt({"stage": "technical"})
        state["tech_approved"] = tech_approved
        if not tech_approved:
            return {"status": "rejected"}

    if not state.get("business_approved"):
        business_approved = interrupt({"stage": "business"})
        state["business_approved"] = business_approved

    return {"status": "approved" if state["business_approved"] else "rejected"}
```

This prevents re-asking for technical approval after it's already been granted.

</details>

---

## 📝 Next Steps

Once you've thought through these questions, we'll move to the implementation phase where you'll build:

1. **Task 1:** SQL query agent that generates queries but requires approval before execution
2. **Task 2:** Add `interrupt()` before query execution and show the query to the user
3. **Task 3:** Implement approval/rejection logic with state updates
4. **Task 4:** Add query modification capability (user can edit before approving)
5. **Task 5:** Implement multi-stage approval workflow with timeout handling

Ready to start coding? Let me know!
