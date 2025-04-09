import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import cv2

class FashionMNISTPredictor:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Fashion MNIST Predictor")
        self.window.geometry("400x500")

        # Load the model
        self.model = load_model('fashion_mnist_model.h5')
        
        # Class names
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Create GUI elements
        self.image_label = tk.Label(self.window)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(self.window, text="Prediction: None", font=('Arial', 14))
        self.result_label.pack(pady=10)

        load_button = tk.Button(self.window, text="Load Image", command=self.load_image)
        load_button.pack(pady=5)

        predict_button = tk.Button(self.window, text="Predict", command=self.predict)
        predict_button.pack(pady=5)

        reset_button = tk.Button(self.window, text="Reset", command=self.reset)
        reset_button.pack(pady=5)

        self.current_image = None
        self.photo = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = image.resize((200, 200))
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)
            self.current_image = file_path

    def predict(self):
        if self.current_image:
            # Load and preprocess image
            img = cv2.imread(self.current_image, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = img / 255.0
            img = np.expand_dims(img, axis=[0, -1])

            # Make prediction
            prediction = self.model.predict(img)
            predicted_class = np.argmax(prediction[0])
            class_name = self.class_names[predicted_class]
            confidence = float(prediction[0][predicted_class])

            self.result_label.config(text=f"Prediction: {class_name}\nConfidence: {confidence:.2%}")

    def reset(self):
        self.image_label.config(image='')
        self.result_label.config(text="Prediction: None")
        self.current_image = None
        self.photo = None

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = FashionMNISTPredictor()
    app.run()
