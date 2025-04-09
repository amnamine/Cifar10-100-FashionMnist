import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import os

class CIFAR10Predictor:
    def __init__(self, root):
        self.root = root
        self.root.title("CIFAR-10 Image Classifier")
        
        # Load the model
        self.model = load_model('cifar10_model.h5')
        
        # Class labels
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # Buttons
        self.load_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=5)
        
        self.predict_button = tk.Button(self.root, text="Predict", command=self.predict)
        self.predict_button.pack(pady=5)
        self.predict_button.config(state='disabled')
        
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset)
        self.reset_button.pack(pady=5)
        
        # Result label
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)
        
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Load and resize image to 32x32
            image = Image.open(file_path)
            image = image.resize((32, 32))
            self.process_image = image
            
            # Display larger version
            display_image = image.resize((200, 200))
            self.photo = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=self.photo)
            
            self.predict_button.config(state='normal')
            self.result_label.config(text="")
            
    def predict(self):
        if hasattr(self, 'process_image'):
            # Prepare image for prediction
            img_array = np.array(self.process_image)
            img_array = img_array.reshape(1, 32, 32, 3)
            img_array = img_array.astype('float32') / 255.0
            
            # Make prediction
            prediction = self.model.predict(img_array)
            predicted_class = self.classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            # Display result
            self.result_label.config(
                text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
            
    def reset(self):
        self.image_label.config(image='')
        self.result_label.config(text="")
        self.predict_button.config(state='disabled')
        if hasattr(self, 'process_image'):
            del self.process_image

if __name__ == "__main__":
    root = tk.Tk()
    app = CIFAR10Predictor(root)
    root.mainloop()
