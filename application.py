from flask import Flask, redirect, url_for, request
from node import LearningNode
import datetime
import random
import numpy as np
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)

model = None
dataset = []   #data      
state_log = [] #log in case something goes terribly wrong
prediction_results = {}  #ds/dt results
predicted_values = {}    #actual predicted values

DT = 0.1
NOISE_SIGMA = 0.5

# --- NODE INITIALIZATION ---
initial_values = [-1, 0, 1, 2, 3]
nodes = []
for i, val in enumerate(initial_values):
    node = LearningNode(node_id=i+1, initial_state=0)
    node.value = val
    nodes.append(node)


def trainModel(dataset):
    data = np.array(dataset)
    X = data[:, :-1]
    y = data[:, -1]
    m = MLPRegressor(hidden_layer_sizes=(5), activation='relu', max_iter=2000, random_state=42)
    m.fit(X, y)
    return m


# --- SNAPSHOT ---
def snapshot():
    state_log.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "states": {node.id: round(node.value, 3) for node in nodes}
    })


# --- HOME ---
@app.route("/", methods=["GET"])
def home():
    global model

    NODE_COUNT = int(request.args.get("n", len(nodes)))
    while len(nodes) < NODE_COUNT:
        node = LearningNode(node_id=len(nodes)+1, initial_state=0)
        node.value = random.uniform(1, 3)
        nodes.append(node)
    if len(nodes) > NODE_COUNT:
        nodes[:] = nodes[:NODE_COUNT]

    if len(dataset) >= 5 and model is None:
        model = trainModel(dataset)

    MAX_DISPLAY = 50
    display_nodes = nodes[:MAX_DISPLAY]

    html = """
    <html><head><style>
        body { font-family: Arial, sans-serif; background-color: #fff0f5; color: #2c4a5a; margin: 20px; }
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
    </style></head><body>

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
    html += "<div style='max-height:400px;overflow-y:auto;width:80%'>"
    html += "<table><tr><th>ID</th><th>State</th><th>Value</th><th>Weight</th><th>Threshold</th><th>Predicted ds/dt</th><th>Predicted Value</th></tr>"
    for n in display_nodes:
        t = (n.state - n.min_state) / (n.max_state - n.min_state)
        r = int(232 + (58  - 232) * t) #red
        g = int(160 + (157 - 160) * t) #green
        b = int(180 + (191 - 180) * t) 
        c = f"rgb({r},{g},{b})"
        pred_val = f"{prediction_results[n.id]:+.4f}" if n.id in prediction_results else "—"
        pred_next = f"{predicted_values[n.id]:.4f}" if n.id in predicted_values else "—"
        html += f"<tr><td>{n.id}</td><td style='background:{c};color:#fff'>{n.state}</td><td>{n.value:.4f}</td><td>{n.weight:.2f}</td><td>{n.threshold:.2f}</td><td style='color:#e8a0b4;font-weight:600'>{pred_val}</td><td style='color:#3a9dbf;font-weight:600'>{pred_next}</td></tr>"
    if len(nodes) > MAX_DISPLAY:
        html += f"<tr><td colspan='6' style='color:#7aa8bc;font-style:italic'>... and {len(nodes)-MAX_DISPLAY} more</td></tr>"
    html += "</table></div>"

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
    html += f"""<div class="section">
<h2>Random Run</h2>
<form action="/random_run_input" method="post">
    Steps: <input type="number" name="steps" value="10" min="1" max="1000">
    <input type="submit" value="Run Random Steps">
</form>
</div>"""

    # smart run
    html += f"""<div class="section">
<h2>Guided Training</h2>
<form action="/smart_run_input" method="post">
    Steps: <input type="number" name="steps" value="10" min="1" max="1000">
    <input type="submit" value="Run Smart Steps">
</form>
</div>"""

    # prediction
    html += "<div class='section'><h2>Predict ds/dt</h2>"
    if model is None:
        html += "<p><i>Run at least 5 steps — model will auto-train.</i></p>"
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
</form>"""
    html += "</div>"

    # model info
    html += "<div class='section'><h2>Model Info</h2>"
    if model:
        html += f"<p>MLPRegressor &nbsp;|&nbsp; Target: ds/dt &nbsp;|&nbsp; Dataset rows: {len(dataset)}</p>"
        if len(dataset) >= 5:
            data = np.array(dataset)
            r2 = model.score(data[:, :-1], data[:, -1])
            html += f"<p>R² on ds/dt: {r2:.4f}</p>"
            html += f"<p>dt = {DT} &nbsp;|&nbsp; noise σ = {NOISE_SIGMA}</p>"
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

@app.route("/predict_inputs", methods=["POST"]) #ds/dt preditction
def predict_inputs():
    global model, prediction_results, predicted_values
    prediction_results = {}
    predicted_values = {}
    if model is None:
        return redirect(url_for('home'))
    for node in nodes:
        try: inp = float(request.form.get(f"n{node.id}", 0))
        except: inp = 0.0
        features = np.array([[node.state, node.value, inp, node.weight, node.threshold]])
        ds_dt = float(model.predict(features)[0])
        prediction_results[node.id] = ds_dt
        predicted_values[node.id] = node.value + DT * ds_dt
    return redirect(url_for('home'))


# --- INPUT STEP ---
@app.route("/input_step", methods=["POST"])
def input_step():
    for node in nodes:
        try: inp = float(request.form.get(f"n{node.id}", 0))
        except: inp = 0.0
        old = (node.state, node.value, node.weight, node.threshold)
        ds_dt = node.step(inputs=[inp], dt=DT, noise_sigma=NOISE_SIGMA, model=model)
        dataset.append([old[0], old[1], inp, old[2], old[3], ds_dt])
    snapshot()
    return redirect(url_for('home'))


# --- RANDOM RUN ---
@app.route("/random_run_input", methods=["POST"])
def random_run_input():
    try: steps = int(request.form.get("steps", 10))
    except: steps = 10
    steps = max(1, min(steps, 1000))
    for _ in range(steps):
        for node in nodes:
            inp = random.uniform(-1, 1)
            old = (node.state, node.value, node.weight, node.threshold)
            ds_dt = node.step(inputs=[inp], dt=DT, noise_sigma=NOISE_SIGMA, model=None)
            dataset.append([old[0], old[1], inp, old[2], old[3], ds_dt])
        snapshot()
    return redirect(url_for('home'))


# --- SMART RUN ---
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
                # model predicts ds/dt at inp=0, then we pick an input to steer
                test = np.array([[node.state, node.value, 0, node.weight, node.threshold]])
                predicted_dsdt = float(model.predict(test)[0])
                inp = predicted_dsdt * 0.5   # nudge input in direction of predicted derivative
            old = (node.state, node.value, node.weight, node.threshold)
            ds_dt = node.step(inputs=[inp], dt=DT, noise_sigma=NOISE_SIGMA, model=model)
            dataset.append([old[0], old[1], inp, old[2], old[3], ds_dt])
        snapshot()
    return redirect(url_for('home'))

@app.route("/train") #retrain 
def train():
    global model
    if len(dataset) >= 5:
        model = trainModel(dataset)
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=True) 