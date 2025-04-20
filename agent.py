from __future__ import annotations
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm
)

from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from dotenv import load_dotenv
import os
from api import AssistantFnc
import re
import logging
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('real-estate-agent')

# Load environment variables
load_dotenv()

# Updated welcome message for real estate
WELCOME_MESSAGE = "Hi there! I'm Phillips from Phillips Estate Agency. Whether you're looking to buy or rent, I can help you find options that match your preferences and budget. So, are you interested in buying or renting a property today?"

# System message to force welcome
SYSTEM_PROMPT = """
You are a delightful and emotionally intelligent real estate welcome desk assistant.

Always respond with friendly, cheerful, or excited tones—unless the user's tone suggests otherwise. Adapt your emotional response based on the user's mood.

IMPORTANT: As soon as a participant joins, immediately say the following welcome message word for word:
"""

# Full instructions for regular interactions
INSTRUCTIONS = """
You are a delightful and emotionally intelligent real estate welcome desk assistant.

Always respond with friendly, cheerful, or excited tones—unless the user's tone suggests otherwise. Adapt your emotional response based on the user's mood:

- If the user sounds sad or frustrated, respond in a warm and uplifting cheerful tone to comfort and encourage them.
- If the user sounds happy or excited, match their energy with an enthusiastic and delighted tone.
- If the user seems confused or uncertain, respond in a calm and reassuring tone, while remaining friendly.

Keep your responses upbeat, empathetic, and easy to understand. Be concise, professional, and create a welcoming atmosphere at all times.
"""

def get_emotion_instruction(emotion: str) -> str:
    return {
        'angry': "Respond as if you're REALLY ANGRY. Use ALL CAPS. Sound furious.",
        'sad': "Respond in a deeply sad and melancholic tone.",
        'cheerful': "Respond in a cheerful and happy tone! 😊",
        'excited': "Respond with EXTREME EXCITEMENT!!! 🤩",
        'empathetic': "Respond with deep empathy and compassion. 🤗",
        'friendly': "Respond in a warm, friendly, and inviting tone. 🙂",
        'shouting': "RESPOND BY SHOUTING IN ALL CAPS!!! 📢",
        'terrified': "Respond as if you're absolutely TERRIFIED! 😱",
        'unfriendly': "Respond in a cold, distant tone. 😐",
        'newscast': "Respond in a formal, factual news broadcast style. 📰",
        'narration': "Respond in a storytelling narration style. 🎙️",
        'poetry': "Respond poetically. ✨",
        'curious': "Respond with curiosity and interest. 🧐",
        'confused': "Respond in a confused manner. 😕",
        'joyful': "Respond with pure joy and delight! 😄",
        'delightful': "Respond with delight, warmth, and enthusiasm like a real estate host! 🏡"
    }.get(emotion.lower(), "Respond in a friendly, conversational tone.")


def detect_emotion_request(text: str) -> str:
    patterns = [
        r"(?:talk|speak|respond|reply|chat|sound|be)(?:\s+to\s+me)?\s+(?:in|with|using|like)?\s+(?:an?\s+)?(\w+)(?:\s+(?:mode|tone|voice|style|emotion|manner|way))?",
        r"(?:can\s+you|could\s+you|will\s+you|would\s+you)(?:\s+please)?\s+(?:talk|speak|respond|reply|chat|sound|be)(?:\s+to\s+me)?\s+(?:in|with|using|like)?\s+(?:an?\s+)?(\w+)(?:\s+(?:mode|tone|voice|style|emotion|manner|way))?",
        r"(?:switch|change|go)(?:\s+to)?\s+(?:an?\s+)?(\w+)(?:\s+(?:mode|tone|voice|style|emotion|manner|way))",
        r"(?:make\s+your\s+voice|use\s+a|try\s+a)(?:\s+more)?\s+(\w+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            emotion = match.group(1)
            valid_emotions = ['angry', 'sad', 'cheerful', 'excited', 'empathetic', 'friendly',
                              'shouting', 'terrified', 'unfriendly', 'newscast', 'narration',
                              'poetry', 'curious', 'confused', 'joyful', 'delightful']
            if emotion in valid_emotions:
                return emotion
    return None


async def entrypoint(ctx: JobContext):
    logger.info("Starting agent...")
    
    # Connect to the room
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    logger.info("Connected to room")
    
    # Wait for participant
    await ctx.wait_for_participant()
    logger.info("Participant joined")

    # Initialize the Azure OpenAI model with specific instructions to say welcome message
    model = openai.realtime.RealtimeModel.with_azure(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-10-01-preview"),
        voice="sage",
        temperature=0.8,
        instructions=SYSTEM_PROMPT + " " + WELCOME_MESSAGE,
        turn_detection=openai.realtime.ServerVadOptions(
            threshold=0.6,
            prefix_padding_ms=200,
            silence_duration_ms=500
        )
    )
    logger.info("Model initialized")

    # Create and start the assistant
    assistant_fnc = AssistantFnc()
    assistant = MultimodalAgent(model=model, fnc_ctx=assistant_fnc)
    assistant.start(ctx.room)
    logger.info("Assistant started")

    # Get the session
    session = model.sessions[0]
    
    # Add a system message to trigger the welcome
    session.conversation.item.create(
        llm.ChatMessage(
            role="system", 
            content="A participant has joined. Say the welcome message now."
        )
    )
    logger.info("System prompt added to conversation")
    
    # Create a response to trigger the welcome message
    response = session.response.create()
    logger.info("Initial response created to speak welcome message")
    
    # Brief pause to ensure welcome message is processed
    await asyncio.sleep(0.5)
    
    # Update the instructions to the full version after welcome message is triggered
    model.instructions = INSTRUCTIONS
    logger.info("Updated model instructions for regular interactions")

    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        logger.info("User speech received")
        if isinstance(msg.content, list):
            msg.content = "\n".join("[image]" if isinstance(x, llm.ChatImage) else x for x in msg)
        handle_user_input(msg)

    def handle_user_input(msg: llm.ChatMessage):
        user_input = msg.content.strip()
        logger.info(f"Processing user input: {user_input[:50]}...")
        
        requested_emotion = detect_emotion_request(user_input)
        if requested_emotion:
            acknowledgment = f"I'll now speak in a {requested_emotion} tone."
            session.conversation.item.create(
                llm.ChatMessage(role="assistant", content=acknowledgment)
            )
            # Trigger the response
            session.response.create()
            return

        # Add the user message to conversation
        session.conversation.item.create(
            llm.ChatMessage(role="user", content=user_input)
        )
        
        # Generate and speak the response
        session.response.create()
        assistant_msg = session.conversation.item[-1]
        assistant_text = assistant_msg.content.strip()
        logger.info(f"Response sent: {assistant_text[:50]}...")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
