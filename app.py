import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_classic.agents import AgentExecutor
from langchain.agents import create_agent


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

#custom tools 
from langchain.tools import tool
import math
#***********************************************************************
@tool
def EMI_Calculator_Tool(loan_amount: float, annual_interest_rate: float, years: int) -> str:
    """Calculates the monthly EMI, total interest, and total payment for a loan.

    Args:
        loan_amount: The principal loan amount in Rupees.
        annual_interest_rate: Annual interest rate in percentage (e.g., 8 for 8%).
        years: Loan tenure in years.
    """
    p = loan_amount
    r = (annual_interest_rate / 100) / 12
    n = years * 12

    if r == 0:
        emi = p / n
    else:
        emi = p * r * (math.pow(1 + r, n)) / (math.pow(1 + r, n) - 1)

    total_payment = emi * n
    total_interest_amount = total_payment - p

    return (
        f"Monthly EMI: ₹{emi:,.2f}\n"
        f"Total Interest Payable: ₹{total_interest_amount:,.2f}\n"
        f"Total Amount Payable: ₹{total_payment:,.2f}"
    )
#*********************************************************************
@tool
def calculate_sip(monthly_investment: float, annual_return_rate: float, years: int) -> str:
    """Calculates the future value of a Systematic Investment Plan (SIP).
    Args:
        monthly_investment: The amount invested monthly (P) in Rupees.
        annual_return_rate: Expected annual return rate in percentage (e.g., 12 for 12%).
        years: The investment period in years.
    """
    P = monthly_investment
    r = (annual_return_rate / 100) / 12  # Monthly return rate
    n = years * 12  # Total months
    
    if r == 0:
        future_value = P * n
    else:
        # Standard Annuity Due calculation for regular monthly SIP schedules
        future_value = P * ((math.pow(1 + r, n) - 1) / r) * (1 + r)

    total_invested = P * n
    wealth_gained = future_value - total_invested
    
    return (f"Total Invested Amount: ₹{total_invested:,.2f}\n"
            f"Estimated Future Value: ₹{future_value:,.2f}\n"
            f"Wealth Gained: ₹{wealth_gained:,.2f}")
#*********************************************************************
@tool
def plan_budget(monthly_income: float) -> str:
    """Suggests a monthly budget allocation using the popular 50-30-20 rule.
    Args:
        monthly_income: The total net monthly income in Rupees.
    """
    needs = monthly_income * 0.50
    wants = monthly_income * 0.30
    savings = monthly_income * 0.20
    
    return (f"Budget Breakdown using the 50-30-20 Rule for an income of ₹{monthly_income:,.2f}:\n"
            f"- Needs (50%): ₹{needs:,.2f} (Rent, groceries, utilities, bills)\n"
            f"- Wants (30%): ₹{wants:,.2f} (Dining out, entertainment, shopping)\n"
            f"- Savings & Debt (20%): ₹{savings:,.2f} (Investments, emergency fund, extra debt payments)")

#*********************************************************************
import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

#collect All Tools
tools=[EMI_Calculator_Tool,calculate_sip,plan_budget]

#initializing LLM
llm=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
        ("You are a customer support assistant"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),])
#**********************************************************************

agent=create_agent( 
    model=llm,
    tools=tools,
    system_prompt="You are a customer support assistant."
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#**********************************************************************


#Streamlit UI application start from here
st.set_page_config(page_title="Financial Assistant Bot", page_icon="💰")
st.title(" Financial Assistant Support Bot")
st.caption("Ask me about EMI calculations, SIP future values, or budget planning!")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous conversation messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Handle new user input
if user_query := st.chat_input("How can I help you with your finances today?"):
    
    # 1. Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # 2. Run agent execution loop
    with st.chat_message("assistant"):
        with st.spinner("Calculating..."):
            try:
                response = agent_executor.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                
                output_text = response["output"]
                st.markdown(output_text)
                
                # 3. Append to history session state
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=output_text))
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

 
