from google import genai
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

chat = client.chats.create(model="gemini-2.5-flash")

print("Gemini chat started. Type quit to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    response = chat.send_message(user_input)

    print("\nGemini:")
    print(response.text)
    print()