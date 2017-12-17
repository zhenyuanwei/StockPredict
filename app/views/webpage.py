from app import app
from flask import render_template
from flask import session
from flask import request

@app.route('/')
def hello_world():
    return 'Hello World!'
