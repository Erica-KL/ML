from flask import Flask, jsonify, redirect, url_for, request
from node import LearningNode
import datetime
import random
import numpy as np
model = None
app = Flask(__name__)

dataset = []

initial_values = [-1, 0, 1, 2, 3] #starting values for nodes
nodes = []
for i, val in enumerate(initial_values):
    node = LearningNode(node_id=i+1, initial_state=0) #initial state
    node.value = val 
    nodes.append(node)

state_log = [] 
@app.route("/", methods=["GET"]) 
def home(): #display directs from JSON in a prettier format.

    html = """
    <html> 
    <head>
        <style>
            table { border-collapse: collapse; width: 70%; }
            th, td { border: 1px solid black; padding: 8px; text-align: center; }
            th { background-color: #ffd7d0; }
        </style>
    </head>
    <body>

    <table>
        <tr><th>Node ID</th><th>State</th><th>Value</th><th>Weight</th><th>Threshold</th></tr>
    """
    for node in nodes:
        green = int((node.state - node.min_state)/(node.max_state - node.min_state)*255)
        red = 255 - green
        color = f"rgb({red},{green},0)" #color coding completely optional but has some visual appeal
        html += f"<tr><td>{node.id}</td><td style='background-color:{color}'>{node.state}</td>"
        html += f"<td>{node.value:.2f}</td><td>{node.weight:.2f}</td><td>{node.threshold:.2f}</td></tr>" #should work 
    html += "</table>"


    html += "<h2>User Inputs</h2>"
    html += "<form action='/input_step' method='post'>"
    for node in nodes:
        html += f"Node {node.id} Input: <input type='number' name='n{node.id}' step='0.1' value='0'><br>"
    html += "<input type='submit' value='Step with Inputs'></form>"


    html += "<h2>State Log (last 10)</h2><ul>"  
    for entry in state_log[-10:]:
        html += f"<li>{entry['time']}: {entry['states']}</li>" #list
    html += "</ul>"
    html += "<h2>Machine Learning Model</h2>"

    if model:
        html += f"<p><b>Coefficients:</b> {model.coef_}</p>"
        html += f"<p><b>Intercept:</b> {model.intercept_}</p>"
    else:
        html += "<p>No model trained yet.</p>"
    html += "<p><a href='/train'>Train Model</a></p>"
    html += "<p><a href='/step'>Step once</a> | <a href='/run/10'>Run 10 steps</a>"
    html += "<p><a href='/auto_step'>Auto Random Step</a></p>"
    html += "</body></html>"
 
    return html #display although I broke the JSON display but in making this it took my ability 

@app.route("/step")
def step():
    for node in nodes:
        node.step()  # adds +1 to state, updates value
    snapshot()
    return redirect(url_for('home'))

@app.route("/input_step", methods=["POST"]) #initiate process stay on home page 
def input_step():
    inputs = []
    for node in nodes:
        val = request.form.get(f"n{node.id}", 0)
        try:
            inputs.append([float(val)])
        except:
            inputs.append([0])
    for node, inp in zip(nodes, inputs):
        oldState = node.state
        oldValue = node.value
        oldWeight = node.weight
        oldThreshold = node.threshold
        node.step(inputs=inp)
        newValue = node.value
        dataset.append([
         oldState, oldValue,inp[0],oldWeight,oldThreshold,newValue]) #keep note of inputs and outputs
        
    snapshot()
    return redirect(url_for('home'))

# Run multiple steps
@app.route("/run/<int:steps>") #run a couple 
def run_steps(steps):
    for _ in range(steps):
        for node in nodes:
            node.step()
    snapshot()
    return redirect(url_for('home'))
from  MLLinReg import trainModel

@app.route("/train")
def train():
    global model

    if len(dataset) < 5:
        return redirect(url_for('home'))

    model = trainModel(dataset)
    return redirect(url_for('home'))


@app.route("/random_run/<int:steps>")
def random_run(steps):
    for _ in range(steps):
        for node in nodes:
            rand_input = random.uniform(-3, 3)

            oldState = node.state
            oldValue = node.value
            oldWeight = node.weight
            oldThreshold = node.threshold

            node.step(inputs=[rand_input])

            newValue = node.value

            dataset.append([
                oldState,
                oldValue,
                rand_input,
                oldWeight,
                oldThreshold,
                newValue
            ])
        snapshot()
    return redirect(url_for('home'))

def snapshot(): #at the bottom a log of everything will appear when a step is run
    state_snapshot = {node.id: node.state for node in nodes}
    state_log.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "states": state_snapshot})
@app.route("/auto_step")
def auto_step():
    for node in nodes:
        rand_input = random.uniform(-3, 3)

        oldState = node.state
        oldValue = node.value
        oldWeight = node.weight
        oldThreshold = node.threshold

        node.step(inputs=[rand_input])

        newValue = node.value

        dataset.append([
            oldState,
            oldValue,
            rand_input,
            oldWeight,
            oldThreshold,
            newValue
        ])

    snapshot()
    return redirect(url_for('home'))
@app.route("/predict")
def predict():
    global model

    if model is None:
        return "Try some random sampling first!"

    if len(dataset) == 0:
        return "Try more random sampling Recommended: at least 10"

    test = np.array([[0, 5, 2, 1.0, 0.5]])
    prediction = model.predict(test)

    return f"Predicted next value: {prediction[0]}"

if __name__ == "__main__":
    app.run(debug=True)