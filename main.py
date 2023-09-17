from flask import Flask, render_template, request, redirect, session
import os
import sentiment_analysis
import json

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/sentiment")
def sentiment():
    return render_template("sentimentpage.html")

@app.route("/find_sentiment", methods=["POST"])
def find_sentiment():
    text = request.form.get("text")

    output = sentiment_analysis.sentiment_analysis(text)

    return json.dumps(output.tolist())



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
