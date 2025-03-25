import json
import os
import datetime
import logging
from typing import List, Dict, Optional
mita
from ollama import Client
from pydantic import BaseModel



# Constants
CHAT_HISTORY_FILE = "chat_history.json"
DEFAULT_MESSAGES = [
    {
        'role': 'user',
        'content': 'Меня зовут Паша, я твой разработчик, мне 17 лет. Разговаривай на русском.',
    }
]
OLLAMA_HOST = "http://94.76.83.82:11434/"  # Consider using environment variables for configuration
MODEL_NAME = "hf.co/wqerrewetw/gemma-3-12b-it-abliterated-Q4_K_M-GGUF:latest"  # Consider using environment variables for configuration


class ReminderArgs(BaseModel):
    task: str
    time: str



# Chat History Management
def load_chat_history() -> List[Dict[str, str]]:
    """Loads chat history from a JSON file."""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        logging.info("No chat history file found. Using default messages.")
        return DEFAULT_MESSAGES
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading chat history: {e}. Using default messages.")
        return DEFAULT_MESSAGES


def save_chat_history(messages: List[Dict[str, str]]) -> None:
    """Saves chat history to a JSON file."""
    try:
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
        logging.info("Chat history saved to file.")
    except IOError as e:
        logging.error(f"Error saving chat history: {e}")





# import torch
# from transformers import AutoModelForSequenceClassification
# from transformers import BertTokenizerFast

# tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment')
# model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)


# def predict(text):
#     inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
#     outputs = model(**inputs)
#     predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
#     predicted = torch.argmax(predicted, dim=1).numpy()
#     return predicted


# positive_stickers = ["😊", "🎉", "👍", "💯"]
# negative_stickers = ["😔", "😟", "😥", "💔"]
# neutral_stickers = ["😐", "🤔", "😶", "🤷"]

# import random

# def analyze_chat_history(messages: List[Dict[str, str]], num_messages: int = 5, probability: float = 0.3) -> Optional[str]:
#     """Анализирует последние сообщения чата и возвращает случайный стикер в зависимости от тональности."""

#     if random.random() > probability: # Случайное условие.
#         return None

#     last_messages = messages[-num_messages:]
#     if len(last_messages) < num_messages:
#         return None

#     sentiments = []
#     for message in last_messages:
#         if message['role'] == 'user': 
#             text = message['content']
#             prediction = predict(text)
#             if prediction == 0:
#                 sentiments.append("Neutral")
#             elif prediction == 1:
#                 sentiments.append("Positive")
#             else:
#                 sentiments.append("Negative")

#     # 3. Агрегация тональности (определяем преобладающую)
#     if not sentiments:
#         return None
    
#     positive_count = sentiments.count("Positive")
#     negative_count = sentiments.count("Negative")
#     neutral_count = sentiments.count("Neutral")

#     if positive_count >= negative_count and positive_count >= neutral_count:
#         stickers = positive_stickers
#     elif negative_count >= positive_count and negative_count >= neutral_count:
#         stickers = negative_stickers
#     else:
#         stickers = neutral_stickers

#     # 4. Выбираем случайный стикер
#     sticker = random.choice(stickers)

#     return sticker



async def main():
    """Main function to run the chat application."""
    client = Client(host=OLLAMA_HOST)
    messages = load_chat_history()

    while True:
        try:
            user_input = input("You: ")
            if not user_input:
                continue

            response = client.chat(
                MODEL_NAME,
                messages=messages + [{'role': 'user', 'content': user_input}],
                stream=False
            )

            assistant_response = response.message.content

            # #  ЗДЕСЬ АНАЛИЗИРУЕМ ИСТОРИЮ ЧАТА
            # sticker = analyze_chat_history(messages + [{'role': 'user', 'content': user_input}])
            # gg = ""
            # if sticker:
            #     gg = f"123334545454 {sticker}"

            print(f"Assistant: {assistant_response}", gg)

            messages.append({'role': 'user', 'content': user_input})
            messages.append({'role': 'assistant', 'content': assistant_response})


            save_chat_history(messages)

        except Exception as e:
            logging.exception("An unexpected error occurred:")
            print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
