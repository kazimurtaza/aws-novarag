#!/usr/bin/env python3
"""NovaRAG CLI - Interactive CLI for querying Pydantic AI documentation."""

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default API URL - can be overridden by env var or CLI arg
DEFAULT_API_URL = os.getenv("NOVARAG_API_URL", "http://localhost:8000")

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class Emoji:
    SEARCH = "🔍"
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    ROCKET = "🚀"
    CHART = "📊"
    CLOCK = "⏱️"
    MONEY = "💰"
    BRAIN = "🧠"
    CHAT = "💬"
    DOCUMENT = "📄"
    TOOL = "🛠️"


class NovaRAGCLI:
    """CLI for NovaRAG service."""

    def __init__(self, api_url: str = DEFAULT_API_URL, verbose: bool = False):
        self.api_url = api_url.rstrip("/")
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _print(self, message: str, color: str = Colors.RESET, emoji: str = ""):
        """Print a colored message with optional emoji."""
        print(f"{color}{emoji} {message}{Colors.RESET}")

    def _print_header(self, text: str):
        """Print a section header."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{text:^60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")

    def _print_stats(self, response_data: dict):
        """Print query statistics."""
        stats = {
            "Latency": f"{response_data.get('latency_ms', 0):,} ms",
            "Input Tokens": f"{response_data.get('input_tokens', 0):,}",
            "Output Tokens": f"{response_data.get('output_tokens', 0):,}",
            "Total Tokens": f"{response_data.get('total_tokens', 0):,}",
            "Estimated Cost": f"${response_data.get('estimated_cost_usd', 0):.6f}",
        }

        tools_used = response_data.get('tools_used', [])

        print(f"\n{Colors.BOLD}{Emoji.CHART} Query Statistics:{Colors.RESET}")
        for key, value in stats.items():
            emoji = {
                "Latency": Emoji.CLOCK,
                "Input Tokens": Emoji.DOCUMENT,
                "Output Tokens": Emoji.DOCUMENT,
                "Total Tokens": Emoji.BRAIN,
                "Estimated Cost": Emoji.MONEY,
            }.get(key, "")
            print(f"  {Colors.DIM}•{Colors.RESET} {key:18}: {Colors.GREEN}{value}{Colors.RESET}")

        # Print tools used in order
        if tools_used:
            print(f"\n{Colors.BOLD}{Emoji.TOOL} Tools Used (in order):{Colors.RESET}")
            for i, tool in enumerate(tools_used, 1):
                # Format tool name: replace underscores with spaces, capitalize
                formatted_tool = tool.replace('_', ' ').title()
                print(f"  {Colors.DIM}{i}.{Colors.RESET} {Colors.CYAN}{formatted_tool}{Colors.RESET}")
        else:
            print(f"\n{Colors.DIM}No tools were used for this query.{Colors.RESET}")

    def _print_answer(self, answer: str):
        """Print the answer with formatting."""
        # Remove <thinking> tags and content between them
        start_tag = "<thinking>"
        end_tag = "</thinking>"

        start_idx = answer.find(start_tag)
        while start_idx != -1:
            end_idx = answer.find(end_tag, start_idx)
            if end_idx != -1:
                answer = answer[:start_idx] + answer[end_idx + len(end_tag):]
            else:
                answer = answer[:start_idx]
            start_idx = answer.find(start_tag)

        clean_answer = answer.strip()

        print(f"\n{Colors.BOLD}{Emoji.CHAT} Answer:{Colors.RESET}")
        print(f"{Colors.BLUE}{'─' * 60}{Colors.RESET}\n")

        # Print the answer directly
        print(clean_answer)

        print(f"\n{Colors.BLUE}{'─' * 60}{Colors.RESET}")

    def health_check(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._print("Service is healthy!", Colors.GREEN, Emoji.SUCCESS)
                if self.verbose:
                    print(f"  Version: {data.get('version', 'unknown')}")
                    print(f"  Database: {data.get('database', 'unknown')}")
                return True
            return False
        except Exception as e:
            self._print(f"Health check failed: {e}", Colors.RED, Emoji.ERROR)
            return False

    def get_stats(self) -> dict:
        """Get overall service statistics."""
        try:
            response = self.session.get(f"{self.api_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self._print(f"Failed to get stats: {e}", Colors.RED, Emoji.ERROR)
        return {}

    def query(self, question: str, show_stats: bool = True) -> dict:
        """Query the NovaRAG service."""
        self._print("Querying NovaRAG...", Colors.BLUE, Emoji.SEARCH)

        start_time = time.time()
        try:
            response = self.session.post(
                f"{self.api_url}/query",
                json={"question": question},
                timeout=120
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                self._print(f"Answer received in {elapsed:.2f}s", Colors.GREEN, Emoji.SUCCESS)

                # Print the answer
                self._print_answer(data.get("answer", "No answer received."))

                # Print stats if requested
                if show_stats:
                    self._print_stats(data)

                return data
            else:
                self._print(
                    f"Query failed with status {response.status_code}: {response.text}",
                    Colors.RED,
                    Emoji.ERROR
                )
                return {}
        except requests.Timeout:
            self._print("Request timed out", Colors.RED, Emoji.ERROR)
            return {}
        except Exception as e:
            self._print(f"Query failed: {e}", Colors.RED, Emoji.ERROR)
            return {}

    def interactive_mode(self):
        """Run the CLI in interactive mode."""
        self._print_header(f"{Emoji.ROCKET} NovaRAG Interactive Mode")

        # Health check first
        if not self.health_check():
            self._print("Service is not available. Exiting.", Colors.RED, Emoji.ERROR)
            return

        # Show overall stats
        stats = self.get_stats()
        if stats:
            print(f"\n{Colors.DIM}Overall Statistics:{Colors.RESET}")
            print(f"  Total Queries: {Colors.CYAN}{stats.get('total_queries', 0):,}{Colors.RESET}")
            print(f"  Avg Latency:   {Colors.CYAN}{stats.get('average_latency_ms', 0):.0f} ms{Colors.RESET}")
            print(f"  Total Cost:    {Colors.GREEN}${stats.get('total_cost_usd', 0):.4f}{Colors.RESET}\n")

        print(f"{Colors.DIM}Type your questions about Pydantic AI below.{Colors.RESET}")
        print(f"{Colors.DIM}Commands: /quit, /stats, /health, /clear{Colors.RESET}\n")

        while True:
            try:
                user_input = input(f"{Colors.BOLD}{Colors.GREEN}nova>{Colors.RESET} ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["/quit", "/exit", "/q"]:
                    self._print("Goodbye!", Colors.CYAN, Emoji.INFO)
                    break
                elif user_input.lower() == "/stats":
                    stats = self.get_stats()
                    if stats:
                        self._print_stats(stats)
                    continue
                elif user_input.lower() == "/health":
                    self.health_check()
                    continue
                elif user_input.lower() == "/clear":
                    os.system("clear" if os.name != "nt" else "cls")
                    continue

                # Regular query
                self.query(user_input)
                print()

            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Interrupted. Type /quit to exit.{Colors.RESET}")
            except EOFError:
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NovaRAG CLI - Query Pydantic AI documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python novarag_cli.py

  # Single query
  python novarag_cli.py -q "How do I install Pydantic AI?"

  # Query with stats disabled
  python novarag_cli.py -q "What is Pydantic AI?" --no-stats

  # Use custom API URL
  python novarag_cli.py -q "How to create an agent?" --url http://localhost:8000

Commands in interactive mode:
  /quit     Exit the CLI
  /stats    Show overall statistics
  /health   Check service health
  /clear    Clear the screen
        """
    )

    parser.add_argument(
        "-q", "--query",
        help="Single query to execute (non-interactive mode)",
        type=str
    )
    parser.add_argument(
        "--url",
        help=f"API URL (default: {DEFAULT_API_URL})",
        type=str,
        default=DEFAULT_API_URL
    )
    parser.add_argument(
        "--no-stats",
        help="Disable statistics display after each query",
        action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="Enable verbose output",
        action="store_true"
    )

    args = parser.parse_args()

    # Create CLI instance
    cli = NovaRAGCLI(api_url=args.url, verbose=args.verbose)

    if args.query:
        # Single query mode
        cli.query(args.query, show_stats=not args.no_stats)
    else:
        # Interactive mode
        cli.interactive_mode()


if __name__ == "__main__":
    main()
