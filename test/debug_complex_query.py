from __future__ import annotations

import time
import traceback

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_rag.chain import create_agentic_chain


def main() -> None:
    c = create_agentic_chain()
    q = "What is the square root of 144 and who founded Google?"
    session_id = "debug_complex_query"

    start = time.time()
    try:
        res = c.invoke(q, session_id=session_id)
        elapsed = time.time() - start
        print("SUCCESS", "elapsed_s=", elapsed)
        print("agents_used=", res.get("agents_used"))
        ans = res.get("answer", "")
        print("answer_head=", ans[:240])
        print("sources_present=", "Sources:" in ans)
    except Exception as e:
        elapsed = time.time() - start
        print("FAILED", "elapsed_s=", elapsed)
        print("error_type=", type(e).__name__)
        print("error_message=", str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()

