from flask import Flask, request
from saramsh_package.saramsh import Saramsh
app = Flask("Saramsh Demo")

@app.route('/')
def hello_world():
    return '<label>Paste the article and its title <br></label><form action="/summarize" method="POST"><br><input name="article"><br><input name="title"><input type="submit" value="Summarize"></form>'
 
@app.route("/summarize", methods=['POST'])
def summarize():
    sm = Saramsh(request.form['article'],request.form['title'])
    sm.summarize()
    return ""+sm.summary