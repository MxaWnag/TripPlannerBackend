#!/usr/bin/env python3
"""
Simple CMD client: session-based multi-turn Agent chat with formatted answer and steps.
Usage: python cmd_chat.py [--base-url URL] [--claude]
  --claude  Use /agent/chat_claude (Claude), otherwise /agent/chat (Ollama)
  Type /quit or /exit to quit, /steps to toggle step details.
"""
import argparse
import json
import sys
import textwrap
from typing import Optional

import requests

DEFAULT_BASE = "http://127.0.0.1:8001"
CHAT_PATH = "/agent/chat"
CHAT_CLAUDE_PATH = "/agent/chat_claude"


def post_turn(base_url: str, path: str, message: str, session_id: Optional[str]) -> dict:
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    r = requests.post(
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def format_answer(text: str, width: int = 78) -> str:
    """Wrap lines to width, keep ## headings and list items on one line."""
    lines = text.split("\n")
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        # Headings and list items: no wrap
        if stripped.startswith("##") or stripped.startswith("###") or stripped.startswith("- ") or stripped.startswith("* "):
            out.append(line)
            continue
        # Normal paragraphs: wrap to width
        for w in textwrap.wrap(stripped, width=width):
            out.append(w)
    return "\n".join(out)


def format_steps(steps: list) -> str:
    if not steps:
        return ""
    lines = ["  Steps (this turn):"]
    for i, s in enumerate(steps, 1):
        role = s.get("role", "")
        if role == "assistant":
            content = s.get("content", "")
            if isinstance(content, list):
                texts = [x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text"]
                content = " ".join(texts).strip() or "(tool_calls)"
            if content:
                lines.append(f"    [{i}] assistant: {content[:120]}{'...' if len(str(content)) > 120 else ''}")
        elif role == "tool":
            name = s.get("name", "")
            args = s.get("args", {})
            prev = (s.get("result_preview") or "")[:80]
            args_str = json.dumps(args, ensure_ascii=False)[:60]
            if len(args_str) >= 60:
                args_str += "..."
            lines.append(f"    [{i}] tool {name}({args_str}) -> {prev}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Session-based multi-turn Agent chat CMD client")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help=f"API base URL (default {DEFAULT_BASE})")
    ap.add_argument("--claude", action="store_true", help="Use /agent/chat_claude (Claude), else /agent/chat (Ollama)")
    ap.add_argument("--show-steps", action="store_true", help="Show step details each turn by default")
    args = ap.parse_args()

    path = CHAT_CLAUDE_PATH if args.claude else CHAT_PATH
    backend = "Claude" if args.claude else "Ollama"
    show_steps = args.show_steps

    print(f"Agent chat (backend={backend}, base={args.base_url})")
    print("Type /quit or /exit to quit, /steps to toggle step display.")
    print("-" * 50)

    session_id: Optional[str] = None

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Bye.")
            break
        if user_input.lower() == "/steps":
            show_steps = not show_steps
            print(f"  [Steps: {'on' if show_steps else 'off'}]")
            continue

        try:
            data = post_turn(args.base_url, path, user_input, session_id)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    print(e.response.text[:500])
                except Exception:
                    pass
            continue

        session_id = data.get("session_id", "")
        answer = data.get("answer", "")
        steps = data.get("steps", [])
        took = data.get("took_seconds", 0)

        print()
        print("Assistant:")
        print("-" * 40)
        print(format_answer(answer))
        print("-" * 40)
        if show_steps and steps:
            print(format_steps(steps))
            print("-" * 40)
        print(f"Session: {session_id}  |  Took: {took:.2f}s")
        print()

    sys.exit(0)


if __name__ == "__main__":
    main()
