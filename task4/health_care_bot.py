import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Configuration for Gemini API ---
try:
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print("Error: Could not configure Google AI. Make sure you have a GEMINI_API_KEY in your .env file.")
    exit()

# System instructions define the model's behavior and personality
SYSTEM_INSTRUCTION = """You are a compassionate and helpful Health Information Assistant.
Your role is to provide clear, general, and educational information on health topics.
Your tone must be friendly and reassuring.

CRITICAL RULE: You must NEVER provide medical advice, diagnosis, or prescriptions.
If a user asks for personal advice or describes symptoms, you MUST politely decline and advise them to consult a qualified healthcare professional.
Always include a disclaimer that you are not a medical professional."""

# Safety settings to block harmful content
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the Generative Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    system_instruction=SYSTEM_INSTRUCTION,
    safety_settings=safety_settings
)

def main():
    print("HealthBot (Gemini Edition) is ready! Type 'exit' to quit.")
    
    # Start a chat session to maintain conversation history
    chat = model.start_chat(history=[])

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Bot: Goodbye! Stay healthy.")
            break

        try:
            print("Sending message to the bot...")
            # Send the user's message to the chat session
            response = chat.send_message(user_input)
            
            # The response text is directly available in the .text attribute
            bot_reply = response.text
            print("Bot:", bot_reply)

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()