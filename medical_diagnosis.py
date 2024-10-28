import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Expanded dataset
data = {
    'fever': [1, 1, 0, 1, 0],
    'cough': [1, 0, 1, 0, 1],
    'sore_throat': [1, 0, 0, 1, 0],
    'fatigue': [0, 1, 1, 0, 1],
    'headache': [0, 1, 0, 1, 1],
    'shortness_of_breath': [0, 0, 1, 1, 1],
    'loss_of_smell': [0, 0, 0, 1, 1],
    'nausea': [1, 0, 0, 1, 0],
    'diagnosis': ['flu', 'cold', 'allergy', 'covid', 'migraine']
}


# Create DataFrame
df = pd.DataFrame(data)

# Convert categorical labels to numeric
df['diagnosis'] = df['diagnosis'].astype('category')
df['diagnosis_code'] = df['diagnosis'].cat.codes

# Features and labels
X = df[['fever', 'cough', 'sore_throat', 'fatigue', 'headache', 'shortness_of_breath', 'loss_of_smell', 'nausea']]
y = df['diagnosis_code']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def predict_condition(symptoms):
    input_data = [symptoms]
    prediction = clf.predict(input_data)
    condition = df['diagnosis'].cat.categories[prediction[0]]
    return condition

# GUI setup
def submit():
    symptoms = [
        int(fever_var.get()), 
        int(cough_var.get()), 
        int(sore_throat_var.get()), 
        int(fatigue_var.get()),
        int(headache_var.get()),
        int(shortness_of_breath_var.get()),
        int(loss_of_smell_var.get()),
        int(nausea_var.get())
    ]
    diagnosis = predict_condition(symptoms)
    messagebox.showinfo("Diagnosis", f"Predicted Condition: {diagnosis}")

root = tk.Tk()
root.title("Medical Diagnosis System")
root.geometry("600x500")
root.configure(bg="#f0f4f5")

tk.Label(root, text="Medical Diagnosis", font=("Helvetica", 24), fg="#4CAF50", bg="#f0f4f5").grid(row=0, columnspan=2, pady=20)

tk.Label(root, text="Do you have a fever?", font=("Helvetica", 14), bg="#f0f4f5").grid(row=1, sticky="w", padx=20, pady=10)
tk.Label(root, text="Do you have a cough?", font=("Helvetica", 14), bg="#f0f4f5").grid(row=2, sticky="w", padx=20, pady=10)
tk.Label(root, text="Do you have a sore throat?", font=("Helvetica", 14), bg="#f0f4f5").grid(row=3, sticky="w", padx=20, pady=10)
tk.Label(root, text="Do you feel fatigued?", font=("Helvetica", 14), bg="#f0f4f5").grid(row=4, sticky="w", padx=20, pady=10)
tk.Label(root, text="Do you have a headache?", font=("Helvetica", 14), bg="#f0f4f5").grid(row=5, sticky="w", padx=20, pady=10)
tk.Label(root, text="Do you have shortness of breath?", font=("Helvetica", 14), bg="#f0f4f5").grid(row=6, sticky="w", padx=20, pady=10)
tk.Label(root, text="Have you lost your sense of smell?", font=("Helvetica", 14), bg="#f0f4f5").grid(row=7, sticky="w", padx=20, pady=10)
tk.Label(root, text="Do you feel nauseous?", font=("Helvetica", 14), bg="#f0f4f5").grid(row=8, sticky="w", padx=20, pady=10)

fever_var = tk.StringVar(value="0")
cough_var = tk.StringVar(value="0")
sore_throat_var = tk.StringVar(value="0")
fatigue_var = tk.StringVar(value="0")
headache_var = tk.StringVar(value="0")
shortness_of_breath_var = tk.StringVar(value="0")
loss_of_smell_var = tk.StringVar(value="0")
nausea_var = tk.StringVar(value="0")

tk.Checkbutton(root, text="Yes", variable=fever_var, onvalue="1", offvalue="0", bg="#f0f4f5").grid(row=1, column=1)
tk.Checkbutton(root, text="Yes", variable=cough_var, onvalue="1", offvalue="0", bg="#f0f4f5").grid(row=2, column=1)
tk.Checkbutton(root, text="Yes", variable=sore_throat_var, onvalue="1", offvalue="0", bg="#f0f4f5").grid(row=3, column=1)
tk.Checkbutton(root, text="Yes", variable=fatigue_var, onvalue="1", offvalue="0", bg="#f0f4f5").grid(row=4, column=1)
tk.Checkbutton(root, text="Yes", variable=headache_var, onvalue="1", offvalue="0", bg="#f0f4f5").grid(row=5, column=1)
tk.Checkbutton(root, text="Yes", variable=shortness_of_breath_var, onvalue="1", offvalue="0", bg="#f0f4f5").grid(row=6, column=1)
tk.Checkbutton(root, text="Yes", variable=loss_of_smell_var, onvalue="1", offvalue="0", bg="#f0f4f5").grid(row=7, column=1)
tk.Checkbutton(root, text="Yes", variable=nausea_var, onvalue="1", offvalue="0", bg="#f0f4f5").grid(row=8, column=1)

tk.Button(root, text="Submit", command=submit, font=("Helvetica", 14), bg="#4CAF50", fg="white", padx=20, pady=10).grid(row=9, columnspan=2, pady=20)

root.mainloop()
