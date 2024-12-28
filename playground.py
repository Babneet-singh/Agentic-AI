from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import phi.api
from phi.playground import Playground
from phi.playground import Playground, serve_playground_app

# Importing Dotenv
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

if groq_api_key:
    print("Groq API Key loaded successfully!")
else:
    print("Failed to load Groq API Key. Check your .env file.")

phi_key=os.getenv("PHI_API_KEY")

if phi_key:
    print("PHI Key loaded successfully!")
else:
    print("Failed to load PHI API Key. Check your .env file.")



web_search_agent=Agent(
    name="Web search Agent",
    role="Search web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instruction=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

##Financial Agent
finance_agent=Agent(
    name="Finance Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True

)


multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)
multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for Reliance", stream=True)

app = Playground(agents=[finance_agent, web_search_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app")