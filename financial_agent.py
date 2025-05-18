from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

# Load your API keys
load_dotenv()

# Initialize Groq model
groq_model = Groq(id="llama3-70b-8192")

# Create a single all-in-one agent with both tools
financial_info_agent = Agent(
    name="Financial Info Agent",
    role="Search for financial data, news, and analysis",
    model=groq_model,
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        ),
        DuckDuckGo
    ],
    instructions=[
        "Use tables to display the financial data.",
        "Always include the source when using web search."
    ],
    show_tool_calls=True,
    markdown=True,
)

# Run the agent with a clear, simple prompt
financial_info_agent.print_response(
    "Provide analyst recommendations, stock fundamentals, and latest news for NVDA."
)
