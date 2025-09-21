import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib
import os

model_path = "mnist_model.pkl"

if not os.path.exists(model_path):
    print("Model not found! Run training script first.")
    exit()

model = joblib.load(model_path)

class DigitRecognizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()

        self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
        self.button_predict.pack()

        self.button_clear = tk.Button(self, text="Clear", command=self.clear)
        self.button_clear.pack()

        self.label_result = tk.Label(self, text="Draw a digit and click Predict")
        self.label_result.pack()

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        r = 15
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def predict_digit(self):
        img = ImageOps.invert(self.image).resize((28, 28))
        img_array = np.array(img).reshape(1, -1) / 255.0
        prediction = model.predict(img_array)
        self.label_result.config(text=f"Predicted Digit: {prediction[0]}")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a digit and click Predict")

if __name__ == "__main__":
    app = DigitRecognizerApp()
    app.mainloop()
