# -*- coding: utf-8 -*-
"""SpeakerVerification.ipynb

"""

# New mp4 or mov to wav
# Install pydub if it's not installed yet.
from google.colab import drive
drive.mount('/content/drive')
!pip install pydub

from pydub import AudioSegment
import os

# Define input and output directories on Google Drive.
input_dir = "" #place file paths here
output_dir = ""

# List all files in the input directory that have a .mov or .mp4 extension.
files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mov', '.mp4'))]

# Sort the list of files alphabetically. (This makes the "first" file be the first alphabetically.)
files.sort()

# Print the files found, for debugging purposes.
print("Found the following video files:")
for f in files:
    print(f)

# Skip the first file and process the remaining ones.
for file in files[1:]:
    # Construct full file paths.
    input_file = os.path.join(input_dir, file)

    # Determine the extension to set the correct format for pydub.
    ext = os.path.splitext(file)[1][1:].lower()  # e.g., "mov" or "mp4"

    # Define the output file name (same base name with a WAV extension).
    base_name = os.path.splitext(file)[0]
    output_file = os.path.join(output_dir, base_name + ".wav")

    # Load the video file using pydub (ffmpeg is used under the hood).
    audio = AudioSegment.from_file(input_file, format=ext)

    # Export the extracted audio to a WAV file.
    audio.export(output_file, format="wav")
    print(f"Extracted audio from '{file}' and saved to '{output_file}'.")

print("Audio extraction complete!")

#Old Diarization
#This is for speach diarisation, only needed if multiple speakers are present in audio clip.
#Ensure correct speaker data is extracted, if not just change speaker name and retry
# Worked well for clip 9 where video length was long and speakers voices were clearer

# jupyter notebook --notebook-dir="C:\Users\cutaj\OneDrive\Desktop\ -> Run CMD as Admin
# This process could take a while without using the GPU
#!pip install pyannote.audio
#!pip install pydub
import os

# Set these environment variables as early as possible
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pyannote.audio import Pipeline
from pydub import AudioSegment

# Path to your extracted audio file (WAV)
audio_path = r"C:\Users\cutaj\OneDrive\Desktop\Speech Recognition Project\LL-20250120-VR-C-54-24-10\videoToAudio\extracted_audio.wav"

# Initialize the diarization pipeline using your Hugging Face token.
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=""  # Token acquired from Hugging Face platform
)

# Apply diarization on the audio file.
diarization = pipeline(audio_path)

# Print diarization results: each segment's start and end times with the assigned speaker label.
print("Diarization results:")
for segment, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker} speaks from {segment.start:.1f}s to {segment.end:.1f}s")

# Determine which speaker label corresponds to the male voice.
# For example, assume the male speaker is labeled "SPEAKER_1" (adjust as needed).
male_speaker_label = "SPEAKER_01"

# Load the full audio using pydub.
full_audio = AudioSegment.from_wav(audio_path)

# Create an empty AudioSegment to hold the male speaker's segments.
male_segments = AudioSegment.empty()

# Loop over the diarization segments and extract those for the male speaker.
for segment, _, speaker in diarization.itertracks(yield_label=True):
    if speaker == male_speaker_label:
        start_ms = int(segment.start * 1000)  # pydub uses milliseconds.
        end_ms = int(segment.end * 1000)
        male_segments += full_audio[start_ms:end_ms]

# Define the output path for the extracted male speaker audio.
male_output_path = r"" #place path here

# Export the extracted male speaker segments to a WAV file.
male_segments.export(male_output_path, format="wav")

print("Male speaker audio extracted!")

# New Diarization Attempt 1 - didnt result so well
# Install necessary packages (if not already installed)
!pip install pyannote.audio pydub

import os
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Set environment variables early
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize the speaker diarization pipeline with your Hugging Face token.
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=""  # Replace with your own token.
)

# Define the directory where your extracted WAV files reside.
input_dir = "" #place path here

# List all WAV files in the input directory.
audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]

if not audio_files:
    print("No WAV files found in the specified folder.")
else:
    print("Found the following WAV files:")
    for f in audio_files:
        print(f)

# Process each audio file.
for file in audio_files:
    audio_path = os.path.join(input_dir, file)
    print(f"\nProcessing file: {audio_path}")

    # Apply the speaker diarization pipeline to the audio file.
    diarization = pipeline(audio_path)

    # Load the full audio using pydub.
    full_audio = AudioSegment.from_wav(audio_path)

    # Create a dictionary to aggregate segments per speaker.
    # We assume each file contains two speakers.
    speakers_audio = {}

    # Loop through each diarization segment.
    # Each segment has a start, end (in seconds) and a speaker label.
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers_audio:
            speakers_audio[speaker] = AudioSegment.empty()
        # Convert seconds to milliseconds for pydub slicing.
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        speakers_audio[speaker] += full_audio[start_ms:end_ms]

    # For consistent output names, sort the speaker keys.
    # (This way the "first" speaker will always be the same if you expect two speakers.)
    for speaker in sorted(speakers_audio.keys()):
        # Remove extension and create a new file name for each speaker.
        base_name = os.path.splitext(file)[0]
        # The output file name is constructed as: original base name + "_" + speaker label + ".wav"
        output_file = os.path.join(input_dir, f"{base_name}_{speaker}.wav")

        # Export the aggregated segments for this speaker to a new WAV file.
        speakers_audio[speaker].export(output_file, format="wav")
        print(f"Extracted audio for {speaker} saved as: {output_file}")

print("\nDiarization and speaker audio extraction complete!")

# Second Version of Diarization
!pip install noisereduce
import os
import librosa
import soundfile as sf
import noisereduce as nr
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Set environment variables early to avoid symlink warnings.
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def reduce_noise(input_path, output_temp_path):
    """
    Loads the audio, applies noise reduction, and saves the denoised audio.
    """
    try:
        data, sr = librosa.load(input_path, sr=None)
        # Apply noise reduction. Adjust parameters (e.g., prop_decrease) as needed.
        reduced_noise = nr.reduce_noise(y=data, sr=sr, prop_decrease=1.0)
        sf.write(output_temp_path, reduced_noise, sr)
        print(f"Noise reduction applied and saved temporary file: {output_temp_path}")
        return output_temp_path
    except Exception as e:
        print(f"Error during noise reduction: {e}")
        # If noise reduction fails, return the original file
        return input_path


def diarize_audio(file_path, pipeline):
    """
    Run the diarization pipeline on the given audio file.
    Returns the diarization result.
    """
    try:
        diarization = pipeline(file_path)
        return diarization
    except Exception as e:
        print(f"Error during diarization of {file_path}: {e}")
        return None


def process_file(file_path, pipeline, output_dir, apply_noise_reduction=True):
    """
    Process a single audio file:
      - Optionally apply noise reduction.
      - Run speaker diarization.
      - Aggregate audio segments per speaker.
      - Export separate WAV files for each speaker.
    """
    print(f"\nProcessing file: {file_path}")

    processed_path = file_path
    # Optional noise reduction: create a temporary denoised file.
    if apply_noise_reduction:
        temp_path = file_path.replace('.wav', '_denoised.wav')
        processed_path = reduce_noise(file_path, temp_path)

    # Run diarization. If it fails, skip file.
    diarization = diarize_audio(processed_path, pipeline)
    if diarization is None:
        print(f"Skipping file due to diarization error: {file_path}")
        return

    # Load full audio using pydub.
    try:
        full_audio = AudioSegment.from_wav(processed_path)
    except Exception as e:
        print(f"Could not load audio file {processed_path}: {e}")
        return

    # Dictionary for storing segments by speaker.
    speakers_audio = {}

    # Iterate through the diarization segments.
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers_audio:
            speakers_audio[speaker] = AudioSegment.empty()
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        speakers_audio[speaker] += full_audio[start_ms:end_ms]

    # Export segments: each speaker's segments are saved as a separate WAV file.
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    for speaker, audio_segment in sorted(speakers_audio.items()):
        output_file = os.path.join(output_dir, f"{base_name}_{speaker}.wav")
        try:
            audio_segment.export(output_file, format="wav")
            print(f"Exported audio for {speaker}: {output_file}")
        except Exception as e:
            print(f"Error exporting {speaker} audio for file {file_path}: {e}")

    # Clean up temporary denoised file if noise reduction was applied.
    if apply_noise_reduction and processed_path != file_path:
        try:
            os.remove(processed_path)
            print(f"Temporary file removed: {processed_path}")
        except Exception as e:
            print(f"Could not remove temporary file {processed_path}: {e}")

    print("Diarization complete for file.")


def main():
    # Initialize the diarization pipeline with your Hugging Face token.
    # Replace YOUR_HF_TOKEN_HERE with your actual token.
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="" #place token
    )

    # Define folder path to your extracted audio files (Google Drive path).
    input_dir = "" #place path
    output_dir = input_dir  # You can change this to a different directory if preferred.

    # List all WAV files in the input folder.
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
    if not audio_files:
        print("No WAV files found in the specified folder.")
        return
    else:
        print("Found the following WAV files:")
        for f in audio_files:
            print(f)

    # Process each file individually.
    for file_name in audio_files:
        file_path = os.path.join(input_dir, file_name)
        process_file(file_path, pipeline, output_dir, apply_noise_reduction=True)


if __name__ == "__main__":
    main()

from google.colab import drive
drive.mount("/content/drive", force_remount=True)
#!pip install speechbrain
import os
import glob
from pathlib import Path
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define your file paths using your Google Drive mount (Linux style in Colab)
reference_folder = r"" #insert path
test_folder = r"" #insert path


# Verify that the reference folder exists:
print("Reference folder exists:", os.path.exists(reference_folder))

# Find all reference audio files (WAV and MP3) in the reference folder.
ref_wav_files = glob.glob(os.path.join(reference_folder, "*.wav"))
ref_mp3_files = glob.glob(os.path.join(reference_folder, "*.mp3"))
ref_files = ref_wav_files + ref_mp3_files
print(f"Found {len(ref_files)} reference audio files.")

# Find all test audio files (WAV and MP3) in the test folder.
test_wav_files = glob.glob(os.path.join(test_folder, "*.wav"))
test_mp3_files = glob.glob(os.path.join(test_folder, "*.mp3"))
test_files = test_wav_files + test_mp3_files
print(f"Found {len(test_files)} test audio files.")

# Initialize the SpeechBrain verifier.
verifier = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/content/pretrained_models/spkrec-ecapa-voxceleb"
)

# Set the threshold to 0.3 (you can adjust this value as needed)
threshold = 0.3

# Prepare lists to store results for plotting.
test_names = []
avg_scores_list = []

# Initialise a variable to count the number of test files flagged as same speaker.
same_speaker_count = 0

# For each test file, compare it against every reference file and average the scores.
for test_file in test_files:
    test_file_path = str(Path(test_file).resolve())
    scores = []
    for ref_file in ref_files:
        ref_file_path = str(Path(ref_file).resolve())
        try:
            score, _ = verifier.verify_files(ref_file_path, test_file_path)
            # Convert score to float if needed.
            score_float = score.item() if hasattr(score, "item") else float(score)
            scores.append(score_float)
        except Exception as e:
            print(f"Error processing file pair ({Path(ref_file).name}, {Path(test_file).name}): {e}")
    if len(scores) == 0:
        print(f"No valid scores computed for test file {Path(test_file).name}")
        continue
    avg_score = np.mean(scores)
    prediction = "Same Speaker" if avg_score >= threshold else "Different Speakers"
    print(f"Test File: {Path(test_file).name}")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Prediction: {prediction}\n")

    # If the prediction is "Same Speaker", increment the count.
    if prediction == "Same Speaker":
        same_speaker_count += 1

    # Store the results for plotting.
    test_names.append(Path(test_file).name)
    avg_scores_list.append(avg_score)

# Check if the majority of the test files were flagged as "Same Speaker"
total_test_files = len(test_files)
same_speaker_percentage = (same_speaker_count / total_test_files) * 100

# If more than 75% of test files are flagged as "Same Speaker", print the threshold and count
if same_speaker_percentage >= 75:
    print(f"\nMajority of test files ({same_speaker_percentage:.2f}%) were flagged as 'Same Speaker'.")
    print(f"Threshold used: {threshold}")
    print(f"Number of files flagged as 'Same Speaker': {same_speaker_count}/{total_test_files}")

# Define which files should be highlighted with a different color.
highlight_files = {"Recording.wav"}#example file

# Create a colour list based on file names.
colors = ['orange' if name in highlight_files else 'skyblue' for name in test_names]

# Plot the results.
plt.figure(figsize=(10, 6))
plt.bar(test_names, avg_scores_list, color=colors)
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
plt.xlabel('Test File Name')
plt.ylabel('Average Score')
plt.title('Speaker Verification Average Scores for Test Files')

# Create custom legend entries.
blue_patch = mpatches.Patch(color='skyblue', label='Speech data belonging to speaker')
orange_patch = mpatches.Patch(color='orange', label='Speech data not belonging to speaker')
plt.legend(handles=[blue_patch, orange_patch], loc='upper right')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cosine Similarity - Didn't work as well
import os
import glob
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Mount Google Drive (ensure you've installed and authenticated if needed)
from google.colab import drive
drive.mount('/content/drive')

# Define file paths for the two folders
reference_folder = r"" #insert paths
test_folder = r"" #insert paths

def get_voice_embedding(audio_file):
    """Load and compute the voice embedding for a given audio file."""
    encoder = VoiceEncoder()
    wav = preprocess_wav(audio_file)
    embedding = encoder.embed_utterance(wav)
    return embedding

# Function to get all .wav and .mp3 files from a folder
def get_audio_files(folder):
    wav_files = glob.glob(os.path.join(folder, "*.wav"))
    mp3_files = glob.glob(os.path.join(folder, "*.mp3"))
    return wav_files + mp3_files

# Get audio files from both folders
ref_audio_paths = get_audio_files(reference_folder)
test_audio_paths = get_audio_files(test_folder)

print(f"Found {len(ref_audio_paths)} reference audio files.")
print(f"Found {len(test_audio_paths)} test audio files.")

# Combine files from both folders, adding a label to each filename for identification
all_audio_paths = []
file_names = []  # For labeling in plots
for path in ref_audio_paths:
    all_audio_paths.append(path)
    file_names.append("Mark: " + os.path.basename(path))
for path in test_audio_paths:
    all_audio_paths.append(path)
    file_names.append("Test: " + os.path.basename(path))

print(f"Total audio files for comparison: {len(all_audio_paths)}")

# Compute embeddings for each file
embeddings = []
for path in all_audio_paths:
    try:
        emb = get_voice_embedding(path)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error processing {path}: {e}")

# Ensure we have at least two valid embeddings
if len(embeddings) < 2:
    raise ValueError("Need at least two valid audio samples to compute similarity.")

embeddings = np.vstack(embeddings)

# Compute the pairwise cosine similarity matrix
sim_matrix = cosine_similarity(embeddings)
n = sim_matrix.shape[0]

# Calculate the average similarity over all unique pairs (excluding self-comparisons)
pairwise_similarities = [sim_matrix[i, j] for i in range(n) for j in range(i+1, n)]
avg_similarity = np.mean(pairwise_similarities)
avg_similarity_percentage = avg_similarity * 100

print("Pairwise Cosine Similarity Matrix:")
print(sim_matrix)
print(f"Average Cosine Similarity: {avg_similarity:.2f}")
print(f"Average Similarity Percentage: {avg_similarity_percentage:.2f}%")

# ----------------------- Plot 1: Heatmap for Pairwise Similarity -----------------------
plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, interpolation='nearest', cmap='viridis')
plt.title("Pairwise Cosine Similarity Heatmap")
plt.xlabel("Audio Sample")
plt.ylabel("Audio Sample")
plt.colorbar(label='Cosine Similarity')
ticks = np.arange(n)
plt.xticks(ticks, file_names, rotation=45, fontsize=8)
plt.yticks(ticks, file_names, fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------- Plot 2: Bar Chart for Average Similarity -----------------------
plt.figure(figsize=(6, 4))
plt.bar(['Average Similarity'], [avg_similarity_percentage], color='green')
plt.ylabel("Similarity Percentage")
plt.ylim(0, 100)
plt.title("Average Voice Similarity Score")
plt.show()

# Cosine Similarity - Didn't work as well - tried with case data only
!pip install resemblyzer

import os
import glob
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Mount Google Drive (ensure you've installed and authenticated if needed)
from google.colab import drive
drive.mount('/content/drive')

# Define file paths for the two folders
reference_folder = r"" #insert paths
test_folder = r"" #insert paths

def get_voice_embedding(audio_file):
    """Load and compute the voice embedding for a given audio file."""
    encoder = VoiceEncoder()
    wav = preprocess_wav(audio_file)
    embedding = encoder.embed_utterance(wav)
    return embedding

# Function to get all .wav and .mp3 files from a folder
def get_audio_files(folder):
    wav_files = glob.glob(os.path.join(folder, "*.wav"))
    mp3_files = glob.glob(os.path.join(folder, "*.mp3"))
    return wav_files + mp3_files

# Get audio files from both folders
ref_audio_paths = get_audio_files(reference_folder)
test_audio_paths = get_audio_files(test_folder)

print(f"Found {len(ref_audio_paths)} reference audio files.")
print(f"Found {len(test_audio_paths)} test audio files.")

# Combine files from both folders, adding a label to each filename for identification
all_audio_paths = []
file_names = []  # For labeling in plots
for path in ref_audio_paths:
    all_audio_paths.append(path)
    file_names.append("Mark: " + os.path.basename(path))
for path in test_audio_paths:
    all_audio_paths.append(path)
    file_names.append("Test: " + os.path.basename(path))

print(f"Total audio files for comparison: {len(all_audio_paths)}")

# Compute embeddings for each file
embeddings = []
for path in all_audio_paths:
    try:
        emb = get_voice_embedding(path)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error processing {path}: {e}")

# Ensure we have at least two valid embeddings
if len(embeddings) < 2:
    raise ValueError("Need at least two valid audio samples to compute similarity.")

embeddings = np.vstack(embeddings)

# Compute the pairwise cosine similarity matrix
sim_matrix = cosine_similarity(embeddings)
n = sim_matrix.shape[0]

# Calculate the average similarity over all unique pairs (excluding self-comparisons)
pairwise_similarities = [sim_matrix[i, j] for i in range(n) for j in range(i+1, n)]
avg_similarity = np.mean(pairwise_similarities)
avg_similarity_percentage = avg_similarity * 100

print("Pairwise Cosine Similarity Matrix:")
print(sim_matrix)
print(f"Average Cosine Similarity: {avg_similarity:.2f}")
print(f"Average Similarity Percentage: {avg_similarity_percentage:.2f}%")

# ----------------------- Plot 1: Heatmap for Pairwise Similarity -----------------------
plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, interpolation='nearest', cmap='viridis')
plt.title("Pairwise Cosine Similarity Heatmap")
plt.xlabel("Audio Sample")
plt.ylabel("Audio Sample")
plt.colorbar(label='Cosine Similarity')
ticks = np.arange(n)
plt.xticks(ticks, file_names, rotation=45, fontsize=8)
plt.yticks(ticks, file_names, fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------- Plot 2: Bar Chart for Average Similarity -----------------------
plt.figure(figsize=(6, 4))
plt.bar(['Average Similarity'], [avg_similarity_percentage], color='green')
plt.ylabel("Similarity Percentage")
plt.ylim(0, 100)
plt.title("Average Voice Similarity Score")
plt.show()