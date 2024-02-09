#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 12:47:35 2024

@author: syed
"""

from flask import Flask
from train import train_model

app = Flask(__name__)
port = 5050

@app.route('/')
def initial():
    return "<p> hello world </p>"

@app.route('/train')
def trainer():
    return train_model()
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=port)