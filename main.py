from flask import Flask

# Create the Flask app
app = Flask(__name__, static_folder='static')

# Import routes after creating the app
from routes import app

@app.route("/")
def hello():
    return "Bienvenue à la prédiction de churn/attrition"

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
