
import argparse
import queue
import threading

import numpy as np
import sherpa_onnx
import sounddevice as sd

import asyncio
from twitchio.ext import commands
import os

def get_device_index_by_keywords(keywords):
    # List all available devices
    devices = sd.query_devices()
    
    # Search for devices by matching keywords
    for i, device in enumerate(devices):
        device_name = device['name'].lower()
        if any(keyword.lower() in device_name for keyword in keywords):
            print('selected device: ' + device_name)
            return i
    
    # Return None if no matching device is found
    return None

# Keywords to search for
keywords = ['VAC', 'Cable', 'Sink']  # List of keywords to match

# Get the index of the device matching any of the keywords
device_index = get_device_index_by_keywords(keywords)

if device_index is not None:
    print(f"Device matching keywords {keywords} found at index {device_index}")
else:
    print(f"No device matching keywords {keywords} found. Will use first!")
    device_index = sd.default.device[1]

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--vits-model",
        type=str,
        help="Path to vits model.onnx",
        required=True,
    )

    parser.add_argument(
        "--vits-lexicon",
        type=str,
        default="",
        help="Path to lexicon.txt",
    )

    parser.add_argument(
        "--vits-tokens",
        type=str,
        default="",
        help="Path to tokens.txt",
        required=True,
    )

    parser.add_argument(
        "--vits-data-dir",
        type=str,
        default="",
        help="""Path to the dict directory of espeak-ng. If it is specified,
        --vits-lexicon and --vits-tokens are ignored""",
        required=True,
    )

    parser.add_argument(
        "--vits-dict-dir",
        type=str,
        default="",
        help="Path to the dict directory for models using jieba",
    )

    parser.add_argument(
        "--tts-rule-fsts",
        type=str,
        default="",
        help="Path to rule.fst",
    )

    parser.add_argument(
        "--sid",
        type=int,
        default=0,
        help="""Speaker ID. Used only for multi-speaker models, e.g.
        models trained using the VCTK dataset. Not used for single-speaker
        models, e.g., models trained using the LJ speech dataset.
        """,
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed. Larger->faster; smaller->slower",
    )

    return parser.parse_args()


testing_key = 'Password12344321'
bot_username =  os.getenv("TWITCH_USER", "")
bot_token =  os.getenv("TWITCH_KEY", None)
channel_name = bot_username # this can be any channel, just don't annoy people :)

write_lock = asyncio.locks.Lock()

sample_rate = None

event = threading.Event()

class CancellationToken:
   def __init__(self):
       self.is_cancelled = False

   def cancel(self):
       self.is_cancelled = True

first_message_time = None


def generated_audio_callback(samples: np.ndarray, progress: float):
    """This function is called whenever max_num_sentences sentences
    have been processed.

    Note that it is passed to C++ and is invoked in C++.

    Args:
      samples:
        A 1-D np.float32 array containing audio samples
    """

    # 1 means to keep generating
    # 0 means to stop generating
    if ct.is_cancelled:
        return 0

    return 1

def main():
    global args
    args = get_args()

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=args.vits_model,
                lexicon=args.vits_lexicon,
                data_dir=args.vits_data_dir,
                dict_dir=args.vits_dict_dir,
                tokens=args.vits_tokens,
            ),
            provider=args.provider,
            debug=args.debug,
            num_threads=args.num_threads,
        ),
        rule_fsts=args.tts_rule_fsts,
        max_num_sentences=1,
    )

    if not tts_config.validate():
        raise ValueError("Please check your config")

    global tts
    tts = sherpa_onnx.OfflineTts(tts_config)

    global sample_rate
    sample_rate = tts.sample_rate

    global ct
    ct = CancellationToken()

    print('TTS engine started, Running twitch bot!')
    bot.run()

# Twitch bot setup
bot = commands.Bot(
    token=bot_token,
    prefix='!',
    initial_channels=[channel_name]
)

def add_tts_message(text):
    #messages.put(text)
    audio = tts.generate(
        text,
        sid=args.sid,
        speed=args.speed,
        callback=generated_audio_callback,
    )

    if len(audio.samples) == 0:
        print("Error!")
        global ct
        ct.cancel()
        return

    sd.play(audio.samples, samplerate=sample_rate, device=device_index)


@bot.event()
async def event_message(message):
    # Check if the message is from one of the excluded bots or starts with !commands
    uname = message._author
    if not uname:
        print('this should never happen but here we are.')
        return
    if uname.name.lower() in ['nightbot', 'streamlabs', 'warpworldbot']:
        return
    if message.content.startswith('!'):
        return
    async with write_lock:
        new = f'{uname.display_name} says {message.content}'
        add_tts_message(new)

# Command to handle searching
@bot.command(name='up')
async def status(ctx):
    user = ctx.author
    await ctx.send(f"Up! @{user.name}")

# Run the bot
if __name__ == '__main__':
    print('Starting TTS, Twitch bot will run shortly!')
    ct = None
    try:
        main()
    except KeyboardInterrupt:
        if ct is not None:
            ct.cancel()
