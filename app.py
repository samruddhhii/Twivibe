import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import joblib
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('agg')
# Step 1: Preprocessing the dataset
df = pd.read_csv(r"Apple-Twitter-Sentiment-DFE.csv", encoding='latin1')
df.dropna(subset=['sentiment:confidence'], inplace=True)

# Step 2: Train a sentiment analysis model
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment:confidence'], test_size=0.2)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

model = svm.SVC()
model.fit(X_train, y_train)

# Step 3: Save the trained model
joblib.dump(model, 'sentiment_model.joblib')

# Step 4: Build a Flask application
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle form submission
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the user input from the form
    tweet = request.form['tweet']
    print(f"tweet: {tweet}")

    # Load the saved model
    loaded_model = joblib.load('sentiment_model.joblib')

    # Perform sentiment analysis on the input
    tweet_features = vectorizer.transform([tweet])
    prediction = loaded_model.predict(tweet_features)[0]
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    # print("safe 0")
    # Generate a bar graph of sentiment distribution in the dataset
    sentiment_counts = df['sentiment:confidence'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    plt.savefig('static/sentiment_graph.png')
    # print("safe 1")
    # Create a confusion matrix
    y_pred = loaded_model.predict(X_test)
    # print("safe 2")
    cm = confusion_matrix(y_test.round(), y_pred)
    # print("safe 3")

    # Render the results template with the prediction, graph, and confusion matrix
    return render_template('results.html', prediction=predicted_label, graph='sentiment_graph.png', confusion_matrix=cm)

# Step 9: Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
  
