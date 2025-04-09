import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import cv2

class CIFAR100Predictor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("CIFAR-100 Image Predictor")
        self.window.geometry("600x500")

        # Load the model
        self.model = load_model('cifar100_model_200.h5')
        
        # Create GUI elements
        self.image_label = tk.Label(self.window)
        self.image_label.pack(pady=10)
        
        self.result_label = tk.Label(self.window, text="Prediction: None", font=('Arial', 12))
        self.result_label.pack(pady=10)
        
        # Buttons
        button_frame = tk.Frame(self.window)
        button_frame.pack(pady=10)
        
        self.load_button = tk.Button(button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(button_frame, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.current_image = None
        self.processed_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = image.resize((200, 200))  # Resize for display
            self.current_image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.current_image)
            
            # Prepare image for prediction
            self.processed_image = cv2.imread(file_path)
            self.processed_image = cv2.resize(self.processed_image, (32, 32))
            self.processed_image = self.processed_image / 255.0

    def predict(self):
        if self.processed_image is not None:
            prediction = self.model.predict(np.expand_dims(self.processed_image, axis=0))
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            self.result_label.configure(text=f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%")
        else:
            self.result_label.configure(text="Please load an image first")

    def reset(self):
        self.image_label.configure(image='')
        self.result_label.configure(text="Prediction: None")
        self.current_image = None
        self.processed_image = None

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = CIFAR100Predictor()
    app.run()
