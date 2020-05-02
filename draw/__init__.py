from flask import Flask,render_template,abort,g
from flask_bootstrap import Bootstrap
import keras
import os

app = Flask('WEB')
app.config['SECRET_KEY']="abcdefghijk"
bootstrap = Bootstrap(app)
app.run(host="")

from draw import error,view