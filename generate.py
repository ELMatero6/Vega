import os
import datetime
import tempfile
from llama_cpp import Llama
from rvc_pipe.rvc_infer import rvc_convert
import pyttsx3
import pygame
import logging
import time  # Added for time tracking
from collections import deque  # Added to keep track of the last 10 messages

logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('llama_cpp').setLevel(logging.CRITICAL)
import warnings

warnings.filterwarnings("ignore")

# Initialize the Llama model with your desired configurations
def initialize_llama():
    llama_model = Llama(
        model_path=".\Models\mistral\mistral-7b-instruct-v0.1.Q4_K_S.gguf",
        device ='cuda:0',
        max_tokens=1024,
        temperature=0.8,
        frequency_penalty=0.5,
        presence_penalty=0.5,
    )
    return llama_model

# Function to generate and save audio with Llama
def generate_and_save_audio(prompt, llama_model, output_dir, pitch=60, rate=155):
    try:
        # Generate text using Llama and track the start time
        start_time = time.time()
        output = llama_model(prompt)
        end_time = time.time()
        
        ai_response = output["choices"][0]["text"]

        # Print the AI's response and generation time
        generation_time = end_time - start_time
        print(f"AI: {ai_response}")
        print(f"Generation Time: {generation_time} seconds")

        # Check if the response contains code
        if '```' in ai_response:
            ai_response_to_read = ai_response.split('```')[0]
        else:
            ai_response_to_read = ai_response

        # Initialize the pyttsx3 engine
        engine = pyttsx3.init()

        # Set the pitch and rate
        engine.setProperty('pitch', pitch)
        engine.setProperty('rate', rate)

        # Save the audio file with a unique name in the output directory as .wav
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_wav_path = os.path.join(output_dir, f"output_{timestamp}.wav")

        # Save the speech audio into a file
        engine.save_to_file(ai_response_to_read, output_wav_path)
        engine.runAndWait()

        return output_wav_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Main loop
def main(output_dir):
    # Initialize the Llama model
    llama_model = initialize_llama()

    # Set the static prompt
    prompt = "You are an advanced AI model, designed to be helpful. Your name is Vega, you were written with the programming language python, and your text to speech code is a pipeline of RVC, You have the ability to write code. Respond Politely and only respond to the users questions. :"

    # Initialize a deque with a max length of 10 to store the last 10 messages
    last_10_messages = deque(maxlen=10)

    while True:
        user_input = input("Q: ")

        if user_input.lower() == 'exit':
            break

        # Add the new message to the deque
        last_10_messages.append(user_input)

        # Combine user input with the static prompt and the last 10 messages
        prompt_with_user_input = f"{prompt} {' '.join(list(last_10_messages))} Q: {user_input} A:"

        # Generate a response using Llama
        filename = generate_and_save_audio(prompt_with_user_input, llama_model, output_dir)

        # Convert the audio using rvc_convert
        print("Converting audio...")
        result = rvc_convert(model_path="./models/vega2.pth", input_path=filename)

        print("Playing audio...")

        # Open the file with a file object
        with open(result, 'rb') as f:
            pygame.mixer.init()
            pygame.mixer.music.load(f)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        # Stop the pygame mixer
        pygame.mixer.music.stop()

if __name__ == "__main__":
    # Define the output directory (point to the temp directory)
    output_dir = tempfile.mkdtemp()
    os.makedirs(output_dir, exist_ok=True)

    main(output_dir)
