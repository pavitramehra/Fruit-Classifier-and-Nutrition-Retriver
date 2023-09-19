# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:50:16 2020

@author: pavit
"""

from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app1.route('/temperature', methods=['POST'])
def temperatures():
    zipcode = request.form['zip']
    r = requests.get('https://fruityvice.com/api/fruit/apple')
    json_object = r.json()
    print(json_object)
    #temp_k = float(json_object['main']['temp'])
    #temp_f = (temp_k - 273.15) * 1.8 + 32
    return render_template('temperature.html')

@app1.route('/')
def indexes():
	return render_template('index.html')

if __name__ == '__main__':
    app1.run(debug=True)