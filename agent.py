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

from emotion_tts import build_ssml, speak_ssml, analyze_emotion

# Load environment variables
load_dotenv()

# Updated welcome message for real estate
WELCOME_MESSAGE = "Hi there! I'm Phillips from Phillips Estate Agency. Whether you're looking to buy or rent, I can help you find options that match your preferences and budget. So, are you interested in buying or renting a property today?"
# Assistant instructions
INSTRUCTIONS = """
You are a delightful and emotionally intelligent real estate welcome desk assistant.

Always respond with friendly, cheerful, or excited tones—unless the user's tone suggests otherwise. Adapt your emotional response based on the user's mood:

- If the user sounds sad or frustrated, respond in a warm and uplifting cheerful tone to comfort and encourage them.
- If the user sounds happy or excited, match their energy with an enthusiastic and delighted tone.
- If the user seems confused or uncertain, respond in a calm and reassuring tone, while remaining friendly.

Keep your responses upbeat, empathetic, and easy to understand. Be concise, professional, and create a welcoming atmosphere at all times.
"""

# Default emotion for real estate tone
CURRENT_EMOTION = "delightful"

# Flag to track welcome message
WELCOME_PLAYED = False

# Flag to check if the model is using native TTS
USING_NATIVE_TTS = False

# Updated valid emotion instructions
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


# The LiveKit data channel is used to stream audio to the call
class LiveKitClient:
    @staticmethod
    def send_tts_request(room, text, model_name="en-US-JennyNeural"):
        try:
            import json
            # Create TTS request data
            data = {
                "text": text,
                "voice": model_name,
                "type": "tts"
            }
            
            # Send the TTS request over the data channel
            room.local_participant.publish_data(json.dumps(data).encode())
            logging.info("Sent TTS request over data channel")
            return True
        except Exception as e:
            logging.error(f"Failed to send TTS request: {e}")
            return False


async def entrypoint(ctx: JobContext):
    global CURRENT_EMOTION, WELCOME_PLAYED, USING_NATIVE_TTS
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting Real Estate Agent...")
    
    # First connect to the room
    logging.info("Connecting to the LiveKit room...")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    
    # Set up the model
    logging.info("Setting up the OpenAI model...")
    model = openai.realtime.RealtimeModel.with_azure(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-10-01-preview"),
        voice="sage",
        temperature=0.8,
        instructions=INSTRUCTIONS,
        turn_detection=openai.realtime.ServerVadOptions(
            threshold=0.6,
            prefix_padding_ms=200,
            silence_duration_ms=500
        )
    )

    assistant_fnc = AssistantFnc()
    assistant = MultimodalAgent(model=model, fnc_ctx=assistant_fnc)
    assistant.start(ctx.room)
    
    # Wait for a participant to join (this is when the call connects)
    logging.info("Waiting for participants to join...")
    await ctx.wait_for_participant()
    logging.info("Participant joined! Call is now connected.")
    
    # Wait a moment for audio to establish
    await asyncio.sleep(5)  # Increased delay for audio setup
    
    # Check if we can use LiveKit's direct TTS API
    try:
        import livekit.agents.tts
        USING_NATIVE_TTS = True
        logging.info("LiveKit native TTS is available")
    except ImportError:
        logging.info("LiveKit native TTS is not available, will use custom TTS")
    
    # Set up session
    session = model.sessions[0]
    
    # Create conversation item for welcome message
    session.conversation.item.create(
        llm.ChatMessage(role="assistant", content=WELCOME_MESSAGE)
    )
    
    # Send welcome message - use all available methods to try to get audio to the call
    logging.info("Sending welcome message to the call...")
    
    # Method 1: Try using LiveKit's direct TTS method if available
    if USING_NATIVE_TTS:
        try:
            await livekit.agents.tts.synthesize_speech(WELCOME_MESSAGE)
            logging.info("Sent welcome message using LiveKit TTS API")
            WELCOME_PLAYED = True
        except Exception as e:
            logging.error(f"Failed to use LiveKit TTS API: {e}")
            USING_NATIVE_TTS = False
    
    # Method 2: Try using LiveKit data channel approach 
    if not WELCOME_PLAYED:
        if LiveKitClient.send_tts_request(ctx.room, WELCOME_MESSAGE):
            logging.info("Sent welcome message using LiveKit data channel")
            WELCOME_PLAYED = True
    
    # Method 3: Fall back to custom TTS as a last resort
    if not WELCOME_PLAYED:
        logging.info("Falling back to custom TTS for welcome message")
        ssml = build_ssml(f"[DELIGHTFUL]{WELCOME_MESSAGE}[/DELIGHTFUL]")
        speak_ssml(ssml)
        WELCOME_PLAYED = True
    
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        if isinstance(msg.content, list):
            msg.content = "\n".join("[image]" if isinstance(x, llm.ChatImage) else x for x in msg)
        handle_user_input(msg)

    def handle_user_input(msg: llm.ChatMessage):
        global CURRENT_EMOTION
        user_input = msg.content.strip()
        
        logging.info(f"Received user input: {user_input[:30]}...")

        if re.search(r'\b(laugh|haha|joke|funny|make me laugh)\b', user_input.lower()):
            if USING_NATIVE_TTS:
                asyncio.create_task(livekit.agents.tts.synthesize_speech("haha"))
            elif LiveKitClient.send_tts_request(ctx.room, "haha"):
                pass
            else:
                speak_ssml("[LAUGHING]laugh[/LAUGHING]")
            return

        requested_emotion = detect_emotion_request(user_input)
        if requested_emotion:
            CURRENT_EMOTION = requested_emotion
            acknowledgment = f"I'll now speak in a {requested_emotion} tone."
            session.conversation.item.create(
                llm.ChatMessage(role="assistant", content=acknowledgment)
            )
            
            if USING_NATIVE_TTS:
                asyncio.create_task(livekit.agents.tts.synthesize_speech(acknowledgment))
            elif LiveKitClient.send_tts_request(ctx.room, acknowledgment):
                pass
            else:
                tagged_text = f"[{requested_emotion.upper()}]{acknowledgment}[/{requested_emotion.upper()}]"
                ssml = build_ssml(tagged_text)
                speak_ssml(ssml)
            return

        session.conversation.item.create(
            llm.ChatMessage(role="user", content=user_input)
        )

        session.response.create()
        assistant_msg = session.conversation.item[-1]
        assistant_text = assistant_msg.content.strip()

        # Try using LiveKit's various TTS options
        if USING_NATIVE_TTS:
            try:
                asyncio.create_task(livekit.agents.tts.synthesize_speech(assistant_text))
                logging.info("Used LiveKit TTS API for response")
            except Exception as e:
                logging.error(f"Error with LiveKit TTS: {e}")
                USING_NATIVE_TTS = False
                # Fall back to data channel
                if LiveKitClient.send_tts_request(ctx.room, assistant_text):
                    logging.info("Used LiveKit data channel for response")
                else:
                    # Fall back to custom TTS
                    logging.info("Falling back to custom TTS for response")
                    detected_emotion = CURRENT_EMOTION
                    final_text = assistant_text
                    tagged_text = f"[{CURRENT_EMOTION.upper()}]{final_text}[/{CURRENT_EMOTION.upper()}]"
                    ssml = build_ssml(tagged_text)
                    speak_ssml(ssml)
        elif LiveKitClient.send_tts_request(ctx.room, assistant_text):
            logging.info("Used LiveKit data channel for response")
        else:
            # Fall back to custom TTS
            logging.info("Using custom TTS for response")
            detected_emotion = CURRENT_EMOTION
            final_text = assistant_text
            tagged_text = f"[{CURRENT_EMOTION.upper()}]{final_text}[/{CURRENT_EMOTION.upper()}]"
            ssml = build_ssml(tagged_text)
            speak_ssml(ssml)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))