import numpy as np
import soundfile as sf

# Parameters for a perfect human vocal pitch
sampling_rate = 44100  # CD quality
duration = 4.0         # 4 seconds
frequency = 150.0      # Perfectly flat pitch (Hz)

# Generate a mathematically perfect sine wave (0% Jitter, 0% Shimmer)
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
pure_tone = 0.5 * np.sin(2 * np.pi * frequency * t)

# Save as a clean wav file
sf.write("perfect_healthy_vocal.wav", pure_tone, sampling_rate)
print("Pristine normal vocal file successfully generated!")