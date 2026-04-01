"""Agentic RAG CLI chatbot - interactive command-line interface."""

from agentic_rag.chain import create_agentic_chain


def print_welcome() -> None:
    print("\n" + "=" * 60)
    print("  Agentic RAG Chatbot (LangGraph + LangSmith)")
    print("=" * 60)
    print("\nAvailable Agents:")
    print("  - PDF Agent: Search internal documents")
    print("  - Web Agent: Search the internet")
    print("  - Calculator Agent: Math calculations")
    print("  - Wikipedia Agent: Factual knowledge")
    print("  - Code Agent: Python code execution")
    print("  - SQL Agent: Database queries")
    print("\nCommands:")
    print("  quit      - Exit the chatbot")
    print("  history   - Show conversation history")
    print("  clear     - Clear conversation history")
    print("  sessions  - List active sessions")
    print("=" * 60 + "\n")


def print_response(result: dict) -> None:
    print(
        f"\n[Used {len(result['agents_used'])} agents "
        f"in {result['iterations']} iteration(s)]"
    )
    print(f"Agents: {', '.join(result['agents_used'])}")
    print("-" * 40)
    print(f"Agent: {result['answer']}")


def main() -> None:
    chain = create_agentic_chain()
    session_id = "cli_session"

    print_welcome()

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            normalized = user_input.lower()
            if normalized == "quit":
                print("\nGoodbye!")
                break

            if normalized == "history":
                messages = chain.get_history(session_id)
                print("\n--- Conversation History ---")
                for message in messages:
                    prefix = "You" if message.type == "human" else "Agent"
                    content = message.content
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"  {prefix}: {content}")
                if not messages:
                    print("  (empty)")
                continue

            if normalized == "clear":
                chain.clear_history(session_id)
                print("History cleared.")
                continue

            if normalized == "sessions":
                sessions = chain.list_sessions()
                print(f"\nActive sessions: {sessions}")
                continue

            result = chain.invoke(user_input, session_id)
            print_response(result)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
