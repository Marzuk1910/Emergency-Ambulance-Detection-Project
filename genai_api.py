import google.generativeai as genai

# 🔑 Your API key
genai.configure(api_key="AIzaSyA8AtkgkGDTmylA5bPZOGo15AkzP9QyleQ")

# ✅ Use correct model from your list
model = genai.GenerativeModel("gemini-2.5-flash")

response = model.generate_content(
    "An ambulance is detected at a traffic signal. Traffic density is medium. Suggest smart signal control action."
)

print("\nAI Response:\n")
print(response.text)