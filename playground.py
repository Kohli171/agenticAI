from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
import phi
phi.api = os.getenv("PHI_API_KEY")

# 🌐 Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Searches the internet and provides answers with cited sources.",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=[
        "Use DuckDuckGo only when a clear query is provided.",
        "Do not invoke the tool unless you understand what to search.",
        "Always show the source of the information.",
        "Use tables where appropriate."
    ],
    show_tool_calls=True,
    markdown=True,
)


# 💹 Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    role="Fetches financial data including prices, fundamentals, and news.",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        )
    ],
    instructions=[
        "Use the YFinance tool to answer financial questions.",
        "Display results in clear tables.",
        "If the ticker symbol is missing, ask for it."
    ],
    show_tool_calls=True,
    markdown=True,
)

# 🧪 Launch Playground App
app = Playground(
    agents=[web_search_agent, finance_agent]
).get_app()

# 🚀 Run Server
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
