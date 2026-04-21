# react_agent.py
# Simple ReAct Agent in ONE file - No frameworks

import os
import json
import re
from openai import OpenAI  # Works with Grok too (xAI API is OpenAI compatible)

# ====================== CONFIG ======================
client = OpenAI(
    base_url="https://api.openai.com/v1",      # Change to "https://api.x.ai/v1" for xAI
    api_key=os.getenv("OPENAI_API_KEY")     # or "OPENAI_API_KEY"
)

MODEL = "gpt-4o-mini"   # or "grok-3-beta", "grok-beta", etc.

# ====================== TOOLS ======================
def calculator(expression: str) -> str:
    """Simple calculator for math expressions"""
    try:
        # Safe evaluation (only basic math)
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def web_search(query: str) -> str:
    """Fake web search tool (replace with real Tavily, Serper, or DuckDuckGo API)"""
    # For demo - in real use, call an actual search API
    print(f"🔍 Searching web for: {query}")
    return f"Search results for '{query}':\n- Top result: AI agents are revolutionizing automation.\n- Another: ReAct pattern was introduced in 2022."

TOOLS = {
    "calculator": calculator,
    "web_search": web_search
}

# Tool descriptions for the LLM
TOOL_DESCRIPTIONS = """
You have access to these tools:
- calculator(expression): Perform math calculations. Example: "calculator(15 * 7 + 42)"
- web_search(query): Search the web for up-to-date information.

Use the exact format for tool calls:
Thought: I need to ...
Action: tool_name
Action Input: argument here

After getting the observation, continue reasoning.
"""

# ====================== SYSTEM PROMPT ======================
SYSTEM_PROMPT = f"""You are a helpful AI assistant that uses the ReAct pattern (Reason + Act).

{TOOL_DESCRIPTIONS}

Always follow this format:
Thought: [your reasoning]
Action: [tool_name or "Final Answer"]
Action Input: [input for tool]   (only if using a tool)

When you have the final answer:
Thought: I now know the answer
Action: Final Answer
Action Input: [clear final answer to the user]
"""

# ====================== AGENT CLASS ======================
class ReActAgent:
    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _call_llm(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            temperature=0.7,
            max_tokens=800
        )
        
        reply = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def run(self, query: str, max_steps: int = 10):
        print(f"\n👤 User: {query}\n")
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # reset for new query

        for step in range(max_steps):
            response = self._call_llm(query if step == 0 else "Continue")

            print(f"🤖 Step {step+1}:\n{response}\n")

            # Parse Action
            action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
            action_input_match = re.search(r"Action Input:\s*(.+)", response, re.IGNORECASE)

            if not action_match:
                continue

            action = action_match.group(1).strip().lower()
            action_input = action_input_match.group(1).strip() if action_input_match else ""

            if action == "final answer":
                final_answer = action_input or response.split("Action Input:")[-1].strip()
                print(f"✅ Final Answer: {final_answer}")
                return final_answer

            # Execute tool
            if action in TOOLS:
                print(f"🛠️  Using tool: {action}({action_input})")
                observation = TOOLS[action](action_input)
                print(f"📝 Observation: {observation}\n")
                
                # Feed observation back to LLM
                self.messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                print(f"⚠️  Unknown action: {action}")

        print("⚠️  Max steps reached.")
        return "I couldn't find a final answer."

# ====================== RUN THE AGENT ======================
if __name__ == "__main__":
    agent = ReActAgent()
    
    while True:
        user_input = input("\nAsk me anything (or type 'quit'): ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if user_input:
            agent.run(user_input)