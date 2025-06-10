import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
import os

class EmailSpamClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Spam Classifier")
        self.root.geometry("600x400")
        
        # Initialize or load the model
        self.model = None
        self.model_path = "email_spam_classifier.joblib"
        
        # Check if model exists, otherwise train a simple one
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.train_default_model()
            self.save_model()
        
        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_label = ttk.Label(self.root, text="Email Spam Classifier", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)
        
        # Instructions
        instructions = ttk.Label(self.root, text="Enter email text below to classify as spam or ham:")
        instructions.pack(pady=5)
        
        # Text input area
        self.text_input = tk.Text(self.root, height=10, width=70, wrap=tk.WORD)
        self.text_input.pack(pady=10, padx=10)
        
        # Classify button
        classify_btn = ttk.Button(self.root, text="Classify Email", command=self.classify_email)
        classify_btn.pack(pady=5)
        
        # Result display
        self.result_frame = ttk.Frame(self.root)
        self.result_frame.pack(pady=10)
        
        self.result_label = ttk.Label(self.result_frame, text="Result: ", font=("Helvetica", 12))
        self.result_label.pack(side=tk.LEFT)
        
        self.classification_label = ttk.Label(self.result_frame, text="", font=("Helvetica", 12, "bold"))
        self.classification_label.pack(side=tk.LEFT)
        
        # Load/Save model buttons (for admin/advanced use)
        admin_frame = ttk.Frame(self.root)
        admin_frame.pack(pady=10)
        
        train_btn = ttk.Button(admin_frame, text="Retrain Model", command=self.retrain_model)
        train_btn.pack(side=tk.LEFT, padx=5)
        
        load_btn = ttk.Button(admin_frame, text="Load Model", command=self.load_model_dialog)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = ttk.Button(admin_frame, text="Save Model", command=self.save_model)
        save_btn.pack(side=tk.LEFT, padx=5)
    
    def classify_email(self):
        email_text = self.text_input.get("1.0", tk.END).strip()
        
        if not email_text:
            messagebox.showwarning("Input Error", "Please enter some email text to classify.")
            return
        
        try:
            prediction = self.model.predict([email_text])[0]
            probability = self.model.predict_proba([email_text]).max()
            
            if prediction == "spam":
                self.classification_label.config(text=f"SPAM ({probability:.2%})", foreground="red")
            else:
                self.classification_label.config(text=f"HAM ({probability:.2%})", foreground="green")
                
        except Exception as e:
            messagebox.showerror("Classification Error", f"An error occurred during classification: {str(e)}")
    
    def train_default_model(self):
        # Simple training data - in a real app, you'd use a proper dataset
        data = {
            'text': [
                "Get free money now!!!", 
                "Hi John, let's meet for lunch tomorrow",
                "You've won a prize! Claim it now!",
                "Meeting reminder: 10am in conference room",
                "Viagra special offer just for you",
                "Please review the attached documents",
                "Limited time offer - 50% off everything!",
                "Your account statement is ready",
                "Nigerian prince needs your help",
                "Project update: everything is on track"
            ],
            'label': [
                "spam", "ham", "spam", "ham", "spam", 
                "ham", "spam", "ham", "spam", "ham"
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Create a simple pipeline with TF-IDF and Naive Bayes
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        
        self.model.fit(df['text'], df['label'])
    
    def save_model(self):
        try:
            joblib.dump(self.model, self.model_path)
            messagebox.showinfo("Success", "Model saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.train_default_model()
    
    def load_model_dialog(self):
        # In a real application, you might want to implement file dialog here
        self.load_model()
        messagebox.showinfo("Info", "Model loaded from default location.")
    
    def retrain_model(self):
        # In a real application, you would load a proper dataset here
        confirm = messagebox.askyesno("Confirm", "This will retrain the model with default data. Continue?")
        if confirm:
            self.train_default_model()
            messagebox.showinfo("Success", "Model retrained with default data.")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmailSpamClassifierApp(root)
    root.mainloop()