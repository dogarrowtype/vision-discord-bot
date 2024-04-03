import os
from dotenv import load_dotenv
import discord
#from discord.ext import commands
#import openai
import asyncio
import logging
#import re
import base64
import requests
import aiohttp
from PIL import Image
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Discord Bot Token
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')

vision_model_url = (f"{OPENAI_BASE_URL}/v1/chat/completions")

# Parse the list of channel IDs from the environment variable and convert it to a set
CHANNEL_IDS = os.getenv('CHANNEL_IDS')
if CHANNEL_IDS:
    CHANNEL_IDS = set(map(int, CHANNEL_IDS.split(',')))
else:
    CHANNEL_IDS = None

# Starting message for image analysis
STARTING_MESSAGE = os.getenv('STARTING_MESSAGE', "What’s in this image? If the image is mostly text, please provide the full text.")

# Max tokens amount for OpenAI ChatCompletion
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 300))

# Message prefix
MESSAGE_PREFIX = os.getenv('MESSAGE_PREFIX', "Image Description:")

# Flag to determine if the bot should reply to image links
REPLY_TO_LINKS = os.getenv('REPLY_TO_LINKS', 'true').lower() == 'true'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Openai compatible vision api url: {vision_model_url}")

# Initialize Discord bot with intents for messages and message content
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot =  discord.Client(intents=intents)

# Ensure the OpenAI API key is set
#openai.api_key = OPENAI_API_KEY
#openai.api_base = OPENAI_BASE_URL

async def describe_image(image_url, message_content):
    if message_content != "":
        IMAGE_PROMPT = (f"Please answer this question about the image. Only output raw information. Follow the question exactly.\nUser question: {message_content}\nPlease repeat the question, then answer it.")
        logger.info(f"Custom message: {IMAGE_PROMPT}")
    else:
        IMAGE_PROMPT = STARTING_MESSAGE
    
    try:
        logger.info("Sending request to the model for image analysis...")
        # Check if the URL is from the allowed domain
        allowed_domain = "cdn.discordapp.com"  # Replace with your desired domain
        if not image_url.startswith(f"https://{allowed_domain}"):
                raise ValueError("Invalid image URL domain")
        # Fetch the image from the URL
        response = requests.get(image_url)
        image_data = response.content
        # Convert the image to PNG format
        image = Image.open(BytesIO(image_data))
        png_buffer = BytesIO()
        image.save(png_buffer, format="PNG")
        png_data = png_buffer.getvalue()
        # Encode the PNG image in base64
        base64_data = base64.b64encode(png_data).decode("utf-8")
        # Send the image to the vision API
        messages = []
        if CUSTOM_QUESTION:  # Replace CUSTOM_QUESTION with your condition
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_data}"},
                    },
                    {"type": "text", "text": IMAGE_PROMPT},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_data}"},
                    },
                ],
            })

    # Send the request to the vision API
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": messages,
            "max_tokens": MAX_TOKENS,
        }
        async with session.post(vision_model_url, json=payload, headers=headers) as response:
                data = await response.json()
                logger.info("Received response from the model.")
                # Extracting and returning the response
                if 'choices' in data:
                    # Extract the text from the first choice
                    first_choice_text = data["choices"][0]["message"]["content"].strip()
            
                    # Split the text into chunks to fit within Discord message character limit
                    max_message_length = 1800  # Discord message character limit
                    description_chunks = [first_choice_text[i:i+max_message_length] for i in range(0, len(first_choice_text), max_message_length)]
            
                    return description_chunks
                else:
                    return ["Failed to obtain a description from the model."]

    except Exception as e:
        logger.error(f"Error analyzing image with model: {e}")
        return ["Error analyzing image with model."]

@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name='Everything 👀'))
    logger.info(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    # Ignore messages sent by the bot and in dms
    if (
        message.author == bot.user
        or message.channel.type == discord.ChannelType.private
    ):
        return

    # Check if no specific channels are specified or if the message is in one of the specified channels
    try:
        if not CHANNEL_IDS or message.channel.id in CHANNEL_IDS:
            # Process attachments if any
            if message.attachments:
                async with message.channel.typing():
                    for attachment in message.attachments:
                        if any(attachment.filename.lower().endswith(ext) for ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']):
                            description_chunks = await describe_image(attachment.url, message.content)
        
                            original_message = None  # Store the original message containing the image attachment
                            
                            # Send each description chunk as a separate message
                            for i, chunk in enumerate(description_chunks):
                                # Split message into multiple parts if exceeds the character limit
                                while chunk:
                                    # Truncate the chunk to fit within the Discord message length limit
                                    truncated_chunk = chunk[:1800]
                                    # Send the message as a reply to the original message
                                    if i == 0:
                                        original_message = await message.reply(f"{MESSAGE_PREFIX} {truncated_chunk}")
                                        logger.info("Sending message to Discord...")
                                        logger.info("Message sent successfully.")
                                    else:
                                        # Send subsequent messages as replies to the original message
                                        await original_message.reply(truncated_chunk)
                                        logger.info("Sending message to Discord...")
                                        logger.info("Message sent successfully.")
                                    # Wait for a short delay before sending the next message to avoid rate-limiting
                                    await asyncio.sleep(1)
                                    chunk = chunk[1800:]
    except Exception as e:
        logger.error(f"Error analyzing image with model: {e}")

# Run the bot
async def main():
    await bot.start(DISCORD_BOT_TOKEN)

asyncio.run(main())
