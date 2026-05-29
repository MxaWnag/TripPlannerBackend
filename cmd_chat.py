#!/usr/bin/env python3
"""
TripPlanner agent chat CLI (Rich + prompt_toolkit, Pywen-style UX).

Usage: python cmd_chat.py [--base-url URL] [--claude] [--show-steps] [--no-stream]
  --claude       Use /agent/chat_claude (Claude), otherwise /agent/chat (Ollama)
  --no-stream    Use blocking JSON /chat instead of NDJSON /chat/stream (no live steps)
  /help          Commands and shortcuts
  /quit          Exit (also: exit, quit, q)
  /steps         Toggle step panels (full detail as the agent runs)
  /stream        Toggle live NDJSON stream (default: on)
  /base-url URL  Change API base without restarting
"""
from trip_cli.main import main

if __name__ == "__main__":
    main()
