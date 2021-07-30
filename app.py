

import flask
from flask import Flask, request, render_template
from flask_cors import CORS
import pickle
import os
from newspaper import Article
import urllib
import joblib
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__,template_folder='templates')

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)
    

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    #print(type(news))
    pred = model.predict([news])
    print(pred)
    #pred = int(pred[0][0])
    return render_template('main.html', y='The news is "{}"'.format(pred[0]))

if __name__=="__main__":
    port = int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
    