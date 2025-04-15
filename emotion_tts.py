import os
import re
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
import io
import wave
import struct

# Load environment variables
load_dotenv()

speech_config = speechsdk.SpeechConfig(
    subscription=os.getenv("AZURE_SPEECH_KEY"),
    region=os.getenv("AZURE_SPEECH_REGION")
)

# Set higher quality audio for better telephone experience
speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
)

# Set voice name
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

try:
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    logging.error(f"Failed to initialize SentimentIntensityAnalyzer: {e}")
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

def analyze_emotion(text: str) -> str:
    emotion_keywords = {
        'angry': ['angry', 'mad', 'furious', 'upset', 'annoyed'],
        'sad': ['sad', 'unhappy', 'depressed'],
        'cheerful': ['cheerful', 'happy', 'glad', 'delighted'],
        'excited': ['excited', 'thrilled', 'eager'],
        'empathetic': ['understand', 'feel for you', 'compassion'],
        'friendly': ['friendly', 'kind', 'nice'],
        'delightful': ['delightful', 'warm', 'welcoming'],
        'joyful': ['joyful', 'ecstatic'],
    }
    emotion_scores = {emotion: 0 for emotion in emotion_keywords}
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            emotion_scores[emotion] += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text.lower()))

    sentiment = sia.polarity_scores(text)
    if max(emotion_scores.values()) > 0:
        return max(emotion_scores, key=emotion_scores.get)
    elif sentiment['compound'] >= 0.5:
        return 'cheerful'
    elif sentiment['compound'] <= -0.5:
        return 'sad'
    elif sentiment['compound'] > 0:
        return 'friendly'
    else:
        return 'default'

def build_ssml(text: str) -> str:
    emotion_to_style = {
        'DEFAULT': {'style': 'friendly', 'degree': 1.0},
        'ANGRY': {'style': 'angry', 'degree': 1.5},
        'SAD': {'style': 'sad', 'degree': 1.0},
        'CHEERFUL': {'style': 'cheerful', 'degree': 1.0},
        'EXCITED': {'style': 'excited', 'degree': 1.5},
        'EMPATHETIC': {'style': 'empathetic', 'degree': 1.0},
        'FRIENDLY': {'style': 'friendly', 'degree': 1.0},
        'DELIGHTFUL': {'style': 'cheerful', 'degree': 1.3},
        'JOYFUL': {'style': 'cheerful', 'degree': 2.0},
        'LAUGHING': {'style': 'cheerful', 'degree': 2.0},
    }

    voice_name = speech_config.speech_synthesis_voice_name or "en-US-JennyNeural"
    
    ssml = (
        f'<speak xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xmlns:mstts="http://www.w3.org/2001/mstts" '
        f'version="1.0" xml:lang="en-US">'
        f'<voice name="{voice_name}">'
    )

    # Add a prosody element for better telephone quality
    ssml += '<prosody rate="0.9" pitch="0">'

    pattern = r'\[(.*?)\](.*?)\[/\1\]'
    last_end = 0
    for match in re.finditer(pattern, text, re.DOTALL):
        if match.start() > last_end:
            default_text = text[last_end:match.start()]
            default_style = emotion_to_style['DEFAULT']
            ssml += f'<mstts:express-as style="{default_style["style"]}" styledegree="{default_style["degree"]}">{default_text}</mstts:express-as>'

        emotion_tag = match.group(1)
        emotion_text = match.group(2)
        content_lines = emotion_text.split('\n\n', 1)
        if len(content_lines) > 1:
            emotion_text = content_lines[1]

        style_info = emotion_to_style.get(emotion_tag.upper(), emotion_to_style['DEFAULT'])
        ssml += f'<mstts:express-as style="{style_info["style"]}" styledegree="{style_info["degree"]}">{emotion_text}</mstts:express-as>'
        last_end = match.end()

    if last_end < len(text):
        default_text = text[last_end:]
        default_style = emotion_to_style['DEFAULT']
        ssml += f'<mstts:express-as style="{default_style["style"]}" styledegree="{default_style["degree"]}">{default_text}</mstts:express-as>'

    # Close prosody tag
    ssml += '</prosody>'
    ssml += '</voice></speak>'
    return ssml

def speak_ssml(ssml: str):
    try:
        logging.info("Generating speech using Azure Text-to-Speech")
        
        # Create an audio output stream
        pull_stream = speechsdk.audio.PullAudioOutputStream()
        audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
        
        # Create the speech synthesizer
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, 
            audio_config=audio_config
        )
        
        # Speak the SSML
        result = speech_synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            logging.error(f"Speech synthesis failed with reason: {result.reason}")
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speechsdk.SpeechSynthesisCancellationDetails(result)
                logging.error(f"Error details: {cancellation_details.error_details}")
            return
        
        # Read the audio data from the pull stream
        audio_buffer = bytes()
        audio_size = 4096
        while True:
            audio_chunk = pull_stream.read(audio_size)
            if not audio_chunk or len(audio_chunk) == 0:
                break
            audio_buffer += audio_chunk
        
        # Convert to the right format for LiveKit
        try:
            # Import here to avoid circular imports
            from livekit.agents import audio as livekit_audio
            
            # Log the audio buffer size
            logging.info(f"Generated audio data of size: {len(audio_buffer)} bytes")
            
            # Play audio through LiveKit (directly to the call)
            logging.info("Sending audio to LiveKit for streaming to call")
            livekit_audio.play_audio_bytes(audio_buffer)
            logging.info("Audio sent to LiveKit successfully")
            
        except (ImportError, AttributeError) as e:
            logging.error(f"Failed to use LiveKit audio: {e}")
            
            # Try another LiveKit audio approach if available
            try:
                from livekit.rtc import audio as rtc_audio
                rtc_audio.play_audio(audio_buffer)
                logging.info("Used rtc.audio fallback")
            except (ImportError, AttributeError) as e2:
                logging.error(f"Failed to use rtc.audio fallback: {e2}")
                
                # Final fallback - play locally
                local_audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
                local_synth = speechsdk.SpeechSynthesizer(
                    speech_config=speech_config,
                    audio_config=local_audio_config
                )
                local_synth.speak_ssml_async(ssml).get()
                logging.warning("Fell back to playing on local speakers")
                
    except Exception as e:
        logging.error(f"Exception in speak_ssml: {str(e)}")