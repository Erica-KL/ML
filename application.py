#importng flask etc... sk learn nueral network.
from flask import Flask, redirect, url_for, request
from node import LearningNode
import datetime
import random
import numpy as np
from sklearn.neural_network import MLPRegressor

model = None #moddel=empty set
app = Flask(__name__) #main for callback

dataset = [] #steps 

initial_values = [-1, 0, 1, 2, 3] #{ inital values of the nodes
nodes = []
for i, val in enumerate(initial_values):
    node = LearningNode(node_id=i+1, initial_state=0)
    node.value = val
    nodes.append(node) #}

state_log = []
prediction_results = {}  # log for predictions


def trainModel(dataset): #{ train model with dataset using relu
    data = np.array(dataset)
    X = data[:, :-1]
    y = data[:, -1]
    m = MLPRegressor(hidden_layer_sizes=(34, 34), activation='relu', max_iter=2000, random_state=42)
    m.fit(X, y)
    return m #} 


def snapshot(): #{ node snapshot for log
    state_snapshot = {node.id: node.state for node in nodes}
    state_log.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "states": state_snapshot
    }) #}


@app.route("/", methods=["GET"]) #main page
def home():
    global model

    NODE_COUNT = int(request.args.get("n", len(nodes))) # node count
    while len(nodes) < NODE_COUNT:
        node = LearningNode(node_id=len(nodes)+1, initial_state=0)
        node.value = random.uniform(-1, 3)
        nodes.append(node)
    if len(nodes) > NODE_COUNT:
        nodes[:] = nodes[:NODE_COUNT]

    if len(dataset) >= 5 and model is None: # auto train 
        model = trainModel(dataset) 

    MAX_DISPLAY = 50 
    display_nodes = nodes[:MAX_DISPLAY]

    html = """
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #fff0f5;
                color: #2c4a5a;
                margin: 20px;
            }
            h1 { color: #3a9dbf; } 
            h2 { color: #e8a0b4; }
            table { border-collapse: collapse; width: 80%; margin-bottom: 20px; }
            th, td { border: 1px solid #f4c4d4; padding: 8px; text-align: center; }
            th { background-color: #fde8f0; color: #c0446a; }
            .section { margin-top: 24px; border-top: 2px solid #f4c4d4; padding-top: 14px; }
            input[type=number] { width: 80px; padding: 4px; }
            input[type=submit] { background-color: #e8a0b4; color: white; padding: 6px 14px; border: none; cursor: pointer; }
            input[type=submit]:hover { background-color: #d4849a; }
            button { background-color: #e8a0b4; color: white; padding: 5px 12px; border: none; cursor: pointer; }
            button:hover { background-color: #d4849a; }
            a { color: #3a9dbf; }
            p { margin: 4px 0; }
            .ni-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
            .ni { display: flex; flex-direction: column; font-size: 0.8rem; color: #7aa8bc; }
            .ni input { width: 80px; }
            .pred { font-family: monospace; font-size: 0.85rem; border-left: 3px solid #e8a0b4; padding-left: 8px; margin: 3px 0; }
        </style>
    </head>
    <body>
    <h1>MLP Neural Network</h1>
    """

    # node count
    html += f"""
    <form method="get">
        Nodes: <input type="number" name="n" value="{NODE_COUNT}" min="1" max="5000">
        <input type="submit" value="Update">
    </form><br>
    """

    # node table
    html += "<h2>Nodes</h2>"
    html += "<table><tr><th>ID</th><th>State</th><th>Value</th><th>Weight</th><th>Threshold</th><th>Predicted</th></tr>"
    for n in display_nodes:
        t = (n.state - n.min_state) / (n.max_state - n.min_state)
        r = int(232 + (58  - 232) * t)
        g = int(160 + (157 - 160) * t)
        b = int(180 + (191 - 180) * t)
        c = f"rgb({r},{g},{b})"
        pred_val = f"{prediction_results[n.id]:.2f}" if n.id in prediction_results else "—"
        html += f"<tr><td>{n.id}</td><td style='background:{c};color:#fff'>{n.state}</td><td>{n.value:.2f}</td><td>{n.weight:.2f}</td><td>{n.threshold:.2f}</td><td style='color:#e8a0b4;font-weight:600'>{pred_val}</td></tr>"
    if len(nodes) > MAX_DISPLAY:
        html += f"<tr><td colspan='5' style='color:#7aa8bc;font-style:italic'>... and {len(nodes)-MAX_DISPLAY} more</td></tr>"
    html += "</table>"

    # manual step
    ni = "".join(
        f'<div class="ni"><label>Node {n.id}</label><input type="number" name="n{n.id}" step="0.1" value="0" class="node-inp"></div>'
        for n in nodes 
    )
    html += f"""<div class="section">
<h2>Manual Step</h2>
Fill all: <input type="number" id="fill-val" step="0.1" value="0">
<button type="button" onclick="document.querySelectorAll('.node-inp').forEach(el=>el.value=document.getElementById('fill-val').value)">Set all</button>
<br><br>
<form action="/input_step" method="post">
    <div class="ni-grid">{ni}</div>
    <input type="submit" value="Step with Inputs">
</form>
</div>"""

    # random run
    html += """<div class="section">
<h2>Random Run</h2>
<p>Random input for N steps.</p>
<form action="/random_run_input" method="post">
    Steps: <input type="number" name="steps" value="10" min="1" max="1000">
    <input type="submit" value="Run Random Steps">
</form>
</div>"""

    # smart run
    html += """<div class="section">
<h2>Guided Training</h2>
<p>70% random, 30% model-guided.</p>
<form action="/smart_run_input" method="post">
    Steps: <input type="number" name="steps" value="10" min="1" max="1000">
    <input type="submit" value="Run Smart Steps">
</form>
</div>"""

    # prediction
    html += "<div class='section'><h2>Prediction</h2>"
    if model is None:
        html += "<p><i>Run at least 5 steps first — model will auto-train.</i></p>"
    else:
        pi = "".join(
            f'<div class="ni"><label>Node {n.id}</label><input type="number" name="n{n.id}" step="0.1" value="0" class="pred-inp"></div>'
            for n in nodes
        )
        html += f"""Fill all: <input type="number" id="pred-fill-val" step="0.1" value="0">
<button type="button" onclick="document.querySelectorAll('.pred-inp').forEach(el=>el.value=document.getElementById('pred-fill-val').value)">Set all</button>
<br><br>
<form action="/predict_inputs" method="post">
    <div class="ni-grid">{pi}</div>
    <input type="submit" value="Predict">
</form>
<p><i>Results appear in the Predicted column of the node table above.</i></p>"""
    html += "</div>"

    # model info
    html += "<div class='section'><h2>Model Info</h2>"
    if model:
        html += f"<p>MLPRegressor &nbsp;|&nbsp; Layers: {len(model.coefs_)+1} &nbsp;|&nbsp; Dataset rows: {len(dataset)}</p>"
        if len(dataset) >= 5:
            data = np.array(dataset)
            r2 = model.score(data[:, :-1], data[:, -1]) # R2 score 
            html += f"<p>R²: {r2:.4f}</p>" 
        html += "<p><a href='/train'>↺ Retrain</a></p>"
    else:
        html += "<p><i>Not trained yet. Run 5+ steps.</i></p>"
    html += "</div>"

    # log
    html += "<div class='section'><h2>State Log (last 10)</h2><ul>"
    for entry in reversed(state_log[-10:]):
        html += f"<li>{entry['time']}: {entry['states']}</li>"
    html += "</ul></div>"

    html += "</body></html>"
    return html


@app.route("/predict_inputs", methods=["POST"])
def predict_inputs():
    global model, prediction_results
    prediction_results = {}
    if model is None:
        return redirect(url_for('home'))
    for node in nodes:
        try: inp = float(request.form.get(f"n{node.id}", 0))
        except: inp = 0.0
        features = np.array([[node.state, node.value, inp, node.weight, node.threshold]])
        pred = float(model.predict(features)[0])
        prediction_results[node.id] = pred
    return redirect(url_for('home'))


@app.route("/input_step", methods=["POST"])
def input_step():
    for node in nodes:
        try: inp = float(request.form.get(f"n{node.id}", 0))
        except: inp = 0.0
        old = (node.state, node.value, node.weight, node.threshold)
        node.step(inputs=[inp])
        dataset.append([old[0], old[1], inp, old[2], old[3], node.value])
    snapshot()
    return redirect(url_for('home'))


@app.route("/random_run_input", methods=["POST"])
def random_run_input():
    try: steps = int(request.form.get("steps", 10))
    except: steps = 10
    steps = max(1, min(steps, 1000))
    for _ in range(steps):
        for node in nodes:
            inp = random.uniform(-10, 10)
            old = (node.state, node.value, node.weight, node.threshold)
            node.step(inputs=[inp])
            dataset.append([old[0], old[1], inp, old[2], old[3], node.value])
        snapshot()
    return redirect(url_for('home'))


@app.route("/smart_run_input", methods=["POST"])
def smart_run_input():
    global model
    try: steps = int(request.form.get("steps", 10))
    except: steps = 10
    steps = max(1, min(steps, 1000))
    for _ in range(steps):
        for node in nodes:
            if model is None or random.random() < 0.7:
                inp = float(np.random.normal(loc=node.state * 0.1, scale=1.0)) 
            else:
                test = np.array([[node.state, node.value, 0, node.weight, node.threshold]]) # test input
                inp = (float(model.predict(test)[0]) - node.value) * 0.5
            old = (node.state, node.value, node.weight, node.threshold)
            node.step(inputs=[inp])
            dataset.append([old[0], old[1], inp, old[2], old[3], node.value])
        snapshot()
    return redirect(url_for('home'))


@app.route("/train") #manual train button
def train():
    global model
    if len(dataset) >= 5: #minimum training data though probably should make it larger
        model = trainModel(dataset)
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True)