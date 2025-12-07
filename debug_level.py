"""
Example showing how to run an Agno agent with verbose debug logging (level 2).

Debug level ranges from 1 (basic) to 2 (detailed). We default to level 2 here.
"""
from agno.agent.agent import Agent
from agno.models.anthropic.claude import Claude
from agno.tools.duckduckgo import DuckDuckGoTools


def main() -> None:
    agent = Agent(
        model=Claude(id="claude-3-5-sonnet-20240620"),
        tools=[DuckDuckGoTools()],
        debug_mode=True,
        debug_level=2,  # detailed debug output
    )

    agent.print_response("What is the current price of Tesla?")
    agent.print_response("What is the current price of Apple?")


if __name__ == "__main__":
    main()


