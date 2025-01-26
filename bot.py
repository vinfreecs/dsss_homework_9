import os

import telebot
import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating


def output_genrated_text(input_message):
    # this func takes user input passes through the model and the response is parsed and returned
    messages = [{"role": "user", "content": input_message}]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    # change the max tokens to get more text
    ans_text = outputs[0]["generated_text"]
    ans_text = ans_text.replace(
        "<|user|>\n" + input_message + "</s>\n<|assistant|>\n", ""
    )
    return ans_text

# the api token is stored in a env variable and is sourced here
BOT_TOKEN = os.environ.get("BOT_TOKEN")
# the bot is created using thekey and i am using the pytelegrambotapi to create the bot
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=["start", "hello"])
# when /start or /hello are typed they get this msg
def send_welcome(message):
    bot.reply_to(
        message, "Hi I am a tinay ai assistant , how can i be of service to you?"
    )

@bot.message_handler(func=lambda msg: True)
# sends the msg from user to the model and gets a response and it is replyed to the user
def echo_all(message):
    ans = output_genrated_text(message.text)
    print(ans)
    bot.reply_to(message, ans)

bot.infinity_polling()
