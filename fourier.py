import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, filedialog, Text, Scrollbar, Button, END
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class AudioAnalyzer:
    def __init__(self):
        self.global_harmonics = None
        self.global_time = None
        self.global_sample_rate = None
        self.analysis_complete = False
        self.data = None

    def analyze_audio(self, filenames, x):
        for filename in filenames:    
            try:
                self.global_sample_rate, self.data = wavfile.read(filename)
                self.global_time = np.linspace(0, len(self.data) / self.global_sample_rate, len(self.data))
                fft_result = np.fft.fft(self.data)
                half_length = len(self.data) //   2
                magnitudes = np.abs(fft_result[:half_length])
                phases = np.angle(fft_result[:half_length])
                peak_bin = np.argmax(magnitudes)
                fundamental_frequency = peak_bin * self.global_sample_rate / len(self.data)
                num_harmonics = min(x, half_length)
                harmonics = []
                for i in range(num_harmonics):
                    start_bin = int(i * half_length / num_harmonics)
                    end_bin = int((i +   1) * half_length / num_harmonics)
                    end_bin = min(end_bin, len(magnitudes))
                    avg_magnitude = np.mean(magnitudes[start_bin:end_bin])
                    avg_phase = np.mean(phases[start_bin:end_bin])
                    freq = i * self.global_sample_rate / len(self.data)
                    harmonics.append((freq, avg_magnitude, avg_phase))
                    
                # Sort the harmonics by magnitude and select the top  10
                top_harmonics = sorted(harmonics, key=lambda x: x[1], reverse=True)[:10]
                self.global_harmonics = top_harmonics
                self.analysis_complete = True
                reconstructed_signal = np.zeros_like(self.global_time)
                for freq, amp, pha in top_harmonics:
                    reconstructed_signal += amp * np.cos(2 * np.pi * freq * self.global_time + pha)

                # Create the plots
                plt.subplot(1,   1,   1)
                plt.plot(self.global_time, self.data)
                plt.text(0.01,   0.9, f"Fundamental Frequency: {fundamental_frequency}", transform=plt.gcf().transFigure)
                
                harmonics_fig = plt.figure(figsize=(10,   6 * len(top_harmonics)))
                for i, (freq, amp, pha) in enumerate(top_harmonics):
                    ax = plt.subplot(len(top_harmonics),   1, i +   1)
                    ax.plot(self.global_time, amp * np.cos(2 * np.pi * freq * self.global_time + pha))
                    title = ax.set_title(f'Harmonic {i+1} - Frequency: {freq} Hz', ha='left', y=1.1)
                    title.set_position([0.05,  1.05])  # Adjust the position of the title
                harmonics_fig.subplots_adjust(top=0.9, bottom=0.2, left=0.2, right=0.9, hspace=3.0)
                plt.show()
                text_area.insert(END, f"Main Signal and 10 most significant Harmonics by magnitude plotted\n")
                return self.data
            except Exception as e:
                text_area.insert(END, f"Error: {e}\n")

    def plot_combined_harmonics(self):
        if not self.analysis_complete:
            text_area.insert(END, "Please deconstruct a WAV file first.\n")
            return
        combined_signal = np.zeros_like(self.global_time)
        for freq, amp, pha in self.global_harmonics:
            combined_signal += amp * np.cos(2 * np.pi * freq * self.global_time + pha)
        if np.allclose(combined_signal,  0):
            text_area.insert(END, "The combined signal is all zeros. Please check the harmonics data.\n")
            return
        normalized_signal = combined_signal / (np.max(np.abs(combined_signal)) +  1e-10)
        plt.figure(figsize=(10,  6))
        plt.plot(self.global_time, normalized_signal)
        plt.title('Combined Harmonics')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()
        text_area.insert(END, f"Combined Harmonics plotted\n")

    def plot_magnitudes(self):
        if not self.analysis_complete:
            text_area.insert(END, "Please deconstruct a WAV file first.\n")
            return
        fft_result = np.fft.fft(self.data)
        magnitudes = np.abs(fft_result)
        frequencies = np.linspace(0, self.global_sample_rate /  2, len(magnitudes))
        plt.figure(figsize=(10,  6))
        plt.plot(frequencies, magnitudes[:len(frequencies)])
        plt.title('Magnitudes')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()
        text_area.insert(END, f"Magnitude plotted\n")

    def plot_phases(self):
        if not self.analysis_complete:
            text_area.insert(END, "Please deconstruct a WAV file first.\n")
            return
        fft_result = np.fft.fft(self.data)
        phases = np.angle(fft_result)
        frequencies = np.linspace(0, self.global_sample_rate /  2, len(phases))
        plt.figure(figsize=(10,  6))
        plt.plot(frequencies, phases[:len(frequencies)])
        plt.title('Phases')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phases')
        plt.grid(True)
        plt.show()
        text_area.insert(END, f"Phases plotted\n")

    def open_files(self):
        filenames = filedialog.askopenfilenames(filetypes=[("WAV files", "*.wav")])
        return list(filenames)

# Create an instance of the AudioAnalyzer class
analyzer = AudioAnalyzer()

def on_button_click():
    filenames = analyzer.open_files()
    if filenames:
        # Prompt the user for the number of harmonics to analyze
        x = simpledialog.askinteger("Input", "Please input number of Harmonics to Analyze and the program will plot 10 harmonics with the most magnitudes")
        if x is not None:  # User entered a valid number
            analyzer.data = analyzer.analyze_audio(filenames, x)
        else:
            text_area.insert(END, "No valid number of harmonics was entered.\n")
    else:
        text_area.insert(END, "No files were selected.\n")

root = tk.Tk()
root.geometry('450x450')
root.title("Audio Analysis Tool")

button = Button(root, text="Select Files", command=on_button_click)
button.pack(pady=10)
magnitudes_button = Button(root, text="Plot Magnitudes", command=analyzer.plot_magnitudes)
magnitudes_button.pack(pady=5)
phases_button = Button(root, text="Plot Phases", command=analyzer.plot_phases)
phases_button.pack(pady=(0,5))
combine_button = Button(root, text="Plot Combined Harmonics", command=analyzer.plot_combined_harmonics)
combine_button.pack(pady=(0,5))

text_area = Text(root, height=10, width=50)
text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = Scrollbar(root, command=text_area.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

text_area['yscrollcommand'] = scrollbar.set

root.mainloop()
