import os
from moviepy.audio.fx.volumex import volumex
from moviepy.editor import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QFileDialog, QMessageBox
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

video_path = ""
audio_path = ""

class VolumeAdjustmentNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(VolumeAdjustmentNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

def train_neural_network(inputs, targets):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

def predict(input_data):
    model.eval()
    with torch.no_grad():
        return model(input_data)

def validate_volume(volume_entry, sv):
    # Add volume validation logic if needed
    pass

def select_video_file():
    file_dialog = QFileDialog()
    return file_dialog.getOpenFileName(None, "Select Video File", "", "Video files (*.mp4 *.avi *.mov)")[0]

def select_audio_file():
    file_dialog = QFileDialog()
    return file_dialog.getOpenFileName(None, "Select Audio File", "", "Audio files (*.mp3 *.wav)")[0]

class AddAudioWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.video_path = ""
        self.audio_path = ""

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.video_button = QPushButton("Add a Video", self)
        self.video_button.clicked.connect(self.get_video_file)
        layout.addWidget(self.video_button)

        self.video_name_label = QLabel("No video selected", self)
        layout.addWidget(self.video_name_label)

        self.audio_button = QPushButton("Add Audio", self)
        self.audio_button.clicked.connect(self.get_audio_file)
        layout.addWidget(self.audio_button)

        self.audio_name_label = QLabel("No audio selected", self)
        layout.addWidget(self.audio_name_label)

        self.volume_label = QLabel("Volume (0 - 100)", self)
        layout.addWidget(self.volume_label)

        self.volume_entry = QLineEdit(self)
        layout.addWidget(self.volume_entry)

        self.render_button = QPushButton("Render", self)
        self.render_button.clicked.connect(self.change_volume)
        layout.addWidget(self.render_button)

        self.setLayout(layout)

    def get_video_file(self):
        filename = select_video_file()
        if filename:
            self.video_path = filename
            video_name = os.path.basename(filename)
            self.video_name_label.setText(video_name)

    def get_audio_file(self):
        filename = select_audio_file()
        if filename:
            self.audio_path = filename
            audio_name = os.path.basename(filename)
            self.audio_name_label.setText(audio_name)

    def change_volume(self):
        if self.video_path and self.audio_path:
            try:
                video = VideoFileClip(self.video_path)
                audio = AudioFileClip(self.audio_path)

                volume = float(self.volume_entry.text()) / 100

                
                input_data = torch.tensor([volume], dtype=torch.float32)
                prediction = predict(input_data)

                adjusted_volume = prediction.item()

       
                audio = audio.volumex(adjusted_volume)

                video = video.set_audio(audio)

                
                output_path = os.path.expanduser("~") + f"/videos/{os.path.basename(self.video_path)}_with_audio.mp4"
                video.write_videofile(output_path)

              
                self.plot_loss_curve()

                QMessageBox.information(self, "Success", "Audio added successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
        else:
            QMessageBox.critical(self, "Error", "Please select a video and an audio file")

    def plot_loss_curve(self):
        # Example Seaborn plot (loss curve)
        sns.set(style="whitegrid")
        plt.plot([0.5, 0.4, 0.3, 0.2, 0.1], label='Training Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    app = QApplication([])
    add_audio_window = AddAudioWindow()
    add_audio_window.show()
    app.exec_()


