import os
import sqlite3
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv(override=True)


class SqlAgentState(TypedDict):
    user_question: str
    generated_sql_query: Optional[str]
    query_type: Optional[str]
    affected_tables: Optional[list]
    risk_level: Optional[str]
    approved: Optional[bool]
    query_result: Optional[list]
    error: Optional[str]
    modified_sql_query: Optional[str]
    approval_requested_at: Optional[str]


llm_model = ChatOpenAI(model_name="gpt-4o", temperature=0)


def setup_database():
    """Create sample database with users and orders tables."""
    conn = sqlite3.connect("hitl_sql.db")
    cursor = conn.cursor()

    # Create users table with signup_date and status
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            signup_date DATE NOT NULL,
            status TEXT DEFAULT 'active'
        )
    """)

    # Create orders table with order_date
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product TEXT NOT NULL,
            amount DECIMAL(10, 2),
            order_date DATE NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Insert sample data (only if tables are empty - idempotent)
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        users = [
            (1, "Alice Johnson", "alice@example.com", "2024-01-15", "active"),
            (2, "Bob Smith", "bob@example.com", "2024-02-20", "active"),
            (3, "Charlie Brown", "charlie@example.com", "2023-12-10", "inactive"),
            (4, "Diana Prince", "diana@example.com", "2024-03-05", "active"),
        ]
        cursor.executemany(
            "INSERT INTO users (id, name, email, signup_date, status) VALUES (?, ?, ?, ?, ?)",
            users,
        )

        orders = [
            (1, 1, "Laptop", 1200.00, "2024-01-20"),
            (2, 1, "Mouse", 25.00, "2024-01-21"),
            (3, 2, "Keyboard", 75.00, "2024-02-25"),
            (4, 4, "Monitor", 300.00, "2024-03-10"),
        ]
        cursor.executemany(
            "INSERT INTO orders (id, user_id, product, amount, order_date) VALUES (?, ?, ?, ?, ?)",
            orders,
        )

    conn.commit()
    conn.close()
    print("✅ Sample database created with users and orders tables")


def generate_sql_query(sql_agent_state: SqlAgentState) -> SqlAgentState:
    """Generate SQL query and extract metadata."""

    system_prompt = """You are a SQL expert. Generate SQLite queries based on user questions.

Database schema:
- users (id, name, email, signup_date, status)
- orders (id, user_id, product, amount, order_date)

Rules:
1. Return ONLY the SQL query, no explanations
2. Use proper SQLite syntax
3. Be specific with WHERE clauses

Example:
User: "Show me all active users"
SQL: SELECT * FROM users WHERE status = 'active'
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=sql_agent_state["user_question"]),
    ]

    response = llm_model.invoke(messages)
    sql_query = response.content.strip()

    # Remove markdown code blocks if present
    if sql_query.startswith("```"):
        lines = sql_query.split("\n")
        sql_query = "\n".join(lines[1:-1])

    # ✅ EXTRACT METADATA WITH CODE (not LLM)
    query_upper = sql_query.upper()

    # Determine query type
    if query_upper.startswith("SELECT"):
        query_type = "SELECT"
        risk_level = "low"
    elif query_upper.startswith("INSERT"):
        query_type = "INSERT"
        risk_level = "medium"
    elif query_upper.startswith("UPDATE"):
        query_type = "UPDATE"
        risk_level = "high"
    elif query_upper.startswith("DELETE"):
        query_type = "DELETE"
        risk_level = "high"
    else:
        query_type = "UNKNOWN"
        risk_level = "high"

    # Extract affected tables (simple string matching)
    affected_tables = []
    if (
        "FROM USERS" in query_upper
        or "INTO USERS" in query_upper
        or "UPDATE USERS" in query_upper
        or "DELETE FROM USERS" in query_upper
        or "USERS WHERE" in query_upper
    ):
        affected_tables.append("users")
    if (
        "FROM ORDERS" in query_upper
        or "INTO ORDERS" in query_upper
        or "UPDATE ORDERS" in query_upper
        or "DELETE FROM ORDERS" in query_upper
        or "ORDERS WHERE" in query_upper
    ):
        affected_tables.append("orders")

    print(f"🤖 Generated SQL: {sql_query}")
    print(f"   Type: {query_type} | Risk: {risk_level} | Tables: {affected_tables}")

    return {
        "generated_sql_query": sql_query,
        "query_type": query_type,
        "affected_tables": affected_tables,
        "risk_level": risk_level,
    }


def approval_node(sql_agent_state: SqlAgentState) -> SqlAgentState:
    """
    Request human approval before executing the SQL query.

    This node:
    1.) Creates a rich payload with the query details for the user to review
    2.) Calls interrupt() to pause the execution and wait for the human approval
    3.) Returns the SqlAgentState with the approval status

    IMPORTANT: Code before interrupt() runs TWICE (idempotent)!
    - First time: Creates payload and pauses
    - Second time: After resume, returns the approval value
    """
    if not sql_agent_state.get("approval_requested_at"):
        sql_agent_state["approval_requested_at"] = datetime.now().isoformat()

    approval_payload = {
        "user_question": sql_agent_state["user_question"],
        "query": sql_agent_state["generated_sql_query"],
        "query_type": sql_agent_state["query_type"],
        "affected_tables": sql_agent_state["affected_tables"],
        "risk_level": sql_agent_state["risk_level"],
    }

    # This pauses execution and waits for Command(resume=...)
    human_approval = interrupt(approval_payload)

    requested_at = datetime.fromisoformat(sql_agent_state.get("approval_requested_at"))
    elapsed_time = datetime.now() - requested_at

    if elapsed_time > timedelta(seconds=60):
        return {
            "approved": False,
            "error": "Approval timeout exceeded (60 seconds)",
        }

    if isinstance(human_approval, dict) and "modified_sql_query" in human_approval:
        return {
            "modified_sql_query": human_approval["modified_sql_query"],
            "approved": True,
        }
    else:
        return {"approved": human_approval}


def execute_sql_query(sql_agent_state: SqlAgentState) -> SqlAgentState:
    """Execute the SQL query if approved."""

    if not sql_agent_state["approved"]:
        error_msg = sql_agent_state.get("error", "Query rejected")
        return {"query_result": None, "error": error_msg}

    print("\n✅ Executing approved query...")

    try:
        conn = sqlite3.connect("hitl_sql.db")
        cursor = conn.cursor()

        query = sql_agent_state.get("modified_sql_query") or sql_agent_state.get(
            "generated_sql_query"
        )

        if sql_agent_state.get("modified_sql_query"):
            print("\n✅ Modified query executed by user")
        else:
            print("\n✅ Generated query executed")

        cursor.execute(query)

        # For SELECT, fetch results
        if sql_agent_state["query_type"] == "SELECT":
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]

            # Format as list of dicts
            formatted_results = [dict(zip(column_names, row)) for row in results]

            print(f"   Retrieved {len(formatted_results)} rows")
            conn.close()
            return {"query_result": formatted_results, "error": None}
        else:
            # For INSERT/UPDATE/DELETE, commit and return affected rows
            conn.commit()
            affected_rows = cursor.rowcount
            conn.close()

            print(f"   Affected {affected_rows} rows")
            return {"query_result": [{"affected_rows": affected_rows}], "error": None}

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {"query_result": None, "error": str(e)}


def create_sql_agent():
    """Create the SQL agent graph with HITL approval"""

    graph = StateGraph(SqlAgentState)

    graph.add_node("generate_sql_query", generate_sql_query)
    graph.add_node("approval", approval_node)
    graph.add_node("execute_sql_query", execute_sql_query)

    graph.add_edge(START, "generate_sql_query")
    graph.add_edge("generate_sql_query", "approval")
    graph.add_edge("approval", "execute_sql_query")
    graph.add_edge("execute_sql_query", END)

    # ✅ Create SqliteSaver with direct connection
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    compiled_graph = graph.compile(checkpointer=checkpointer)

    return compiled_graph  # Just return the graph


def main():
    """Entry point for the SQL agent"""

    print("=" * 80)
    print("🧪 SQL Agent with Human-in-the-Loop")
    print("=" * 80)
    print("💡 Type 'exit' or 'quit' to stop")
    print("=" * 80)

    # Setup database (only once)
    setup_database()

    # Create agent (only once)
    agent = create_sql_agent()

    query_count = 0

    while True:
        # Get user question
        print("\n" + "─" * 80)
        user_question = input(
            "\n📝 Enter your SQL question (or 'exit' to quit): "
        ).strip()

        if user_question.lower() in ["exit", "quit", "q"]:
            print("\n� Goodbye!\n")
            break

        if not user_question:
            print("⚠️  Please enter a question!")
            continue

        query_count += 1

        # Use unique thread_id for each query
        config = {"configurable": {"thread_id": f"query-{query_count}"}}

        # Initial invoke - will pause at interrupt()
        result = agent.invoke({"user_question": user_question}, config=config)

        # Check if interrupted
        if "__interrupt__" in result:
            print("\n" + "=" * 80)
            print("⏸️  EXECUTION PAUSED - APPROVAL REQUIRED")
            print("=" * 80)

            # Access the interrupt payload
            interrupt_data = result["__interrupt__"][0].value

            print("\n📋 Approval Request:")
            print(f"   Question: {interrupt_data['user_question']}")
            print(f"   Generated SQL: {interrupt_data['query']}")
            print(f"   Query Type: {interrupt_data['query_type']}")
            print(f"   Risk Level: {interrupt_data['risk_level']}")
            print(f"   Affected Tables: {interrupt_data['affected_tables']}")

            print("\n" + "─" * 80)

            # Ask user for approval
            approval = input("👤 Approve this query? (yes/no/edit): ").strip().lower()

            if approval in ["yes", "y"]:
                print("\n✅ Query approved - executing...")
                # Resume with approval
                final_result = agent.invoke(Command(resume=True), config=config)
            elif approval in ["edit", "e"]:
                edited_sql = input("\n📝 Enter the modified SQL query: ").strip()
                print("\n📝 Query edited - re-executing...")
                # Resume with edit
                final_result = agent.invoke(
                    Command(resume={"modified_sql_query": edited_sql}),
                    config=config,
                )
            else:
                print("\n❌ Query rejected - not executing")
                # Resume with rejection
                final_result = agent.invoke(Command(resume=False), config=config)

            print("\n" + "=" * 80)
            print("✅ EXECUTION COMPLETED")
            print("=" * 80)

            # Print results
            if final_result.get("error"):
                print(f"\n❌ Error: {final_result['error']}")
            elif final_result.get("query_result"):
                print("\n📊 Query Results:")
                for row in final_result["query_result"]:
                    print(f"   {row}")
        else:
            print("\n⚠️  No interrupt detected - something went wrong!")

    print("\n" + "=" * 80)
    print(f"🎉 Completed {query_count} queries!")
    print("=" * 80)


if __name__ == "__main__":
    main()
