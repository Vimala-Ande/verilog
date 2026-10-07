import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# Get Gemini API key from Render Environment Variables
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)

# Generate Verilog HDL
def generate_verilog(input_data):
    # LangServe Playground sends input as a dictionary.
    if isinstance(input_data, dict):
        task = input_data.get("task", "")
    else:
        task = str(input_data)

    if not task.strip():
        return "Please enter a Verilog design task."

    prompt = f"""
You are a Verilog HDL code generator.

Design task:
{task}

Generate the complete Verilog HDL code for the above task.

Rules:
1. Output ONLY Verilog code.
2. Do NOT give explanations.
3. Do NOT use Markdown.
4. Do NOT use ``` code fences.
5. Use standard Verilog HDL syntax.
6. Include a complete module.
7. Include all required inputs and outputs.
8. Make the code synthesizable.
9. Keep the code simple and correct.
10. Do not add unnecessary code.
"""

    response = llm.invoke(prompt)

    content = response.content

    # Handle different Gemini response formats
    if isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text += item["text"]
            else:
                text += str(item)
        content = text

    content = str(content).strip()

    # Remove accidental Markdown code fences
    if content.startswith("```verilog"):
        content = content[len("```verilog"):].strip()

    if content.startswith("```"):
        content = content[3:].strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return content


# Convert the Python function into a LangChain Runnable
formatted_agent_chain = RunnableLambda(generate_verilog)

# FastAPI application
app = FastAPI(
    title="Verilog HDL Generator",
    description="Generate Verilog HDL code using Gemini.",
    version="1.0.0"
)

# IMPORTANT:
# This creates the LangServe route /agent
# Therefore the Playground URL becomes:
# /agent/playground/
add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


@app.get("/")
def home():
    return {
        "message": "Verilog HDL Generator is running.",
        "playground": "/agent/playground/"
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
