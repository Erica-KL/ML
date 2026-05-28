from flask import Flask, redirect, url_for, request
from node import LearningNode
import datetime
import random
import json
import numpy as np
import torch
import torch.nn as nn

app = Flask(__name__)

model = None
dataset = []   #data rows: [T, S, eta1, eta2_eff, eta3, inp, next_T, next_S]
state_log = [] #log in case something goes terribly wrong
prediction_results = {}  #predicted next_T, next_S from model, keyed by node id
DT = 0.1 #time step NECCESSITY for stommel
NOISE_SIGMA = 0.5 #noise represents real ocean variability
NODE_COUNT = 50 #fixed number of sensor nodes

# use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- STOMMELNET ---
# predicts next_T, next_S from 6 stommel features: T, S, eta1, eta2_eff, eta3, inp
# targets are clean RK4 values — noise is external forcing on state, not in training signal
class StommelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),  #6 inputs: T, S, eta1, eta2_eff, eta3, inp
            nn.ReLU(),
            nn.Linear(64, 64), #hidden layer
            nn.ReLU(),
            nn.Linear(64, 32), #hidden layer
            nn.ReLU(),
            nn.Linear(32, 2)   #output: next_T, next_S
        )

    def forward(self, x): #forward pass
        return self.net(x) #shape [batch, 2]


def stommel_residual(X_batch, pred): #physics residual — penalize predictions that violate Stommel equations
    # features: [T, S, eta1, eta2_eff, eta3, inp]
    T        = X_batch[:, 0]
    S        = X_batch[:, 1]
    eta1     = X_batch[:, 2]
    eta2_eff = X_batch[:, 3]
    eta3     = X_batch[:, 4]
    psi      = T - S #Ψ = T - S
    dT_dt    = eta1     - T * (1.0 + torch.abs(psi)) #stommels first box
    dS_dt    = eta2_eff - S * (eta3 + torch.abs(psi)) #stommels second box
    true_next_T = T + DT * dT_dt #what Stommel says next T should be
    true_next_S = S + DT * dS_dt #what Stommel says next S should be
    pred_T = pred[:, 0] #model predicted next T
    pred_S = pred[:, 1] #model predicted next S
    return torch.mean((pred_T - true_next_T) ** 2 + (pred_S - true_next_S) ** 2) #MSE against physics


def trainModel(dataset): #train StommelNet with PINN loss — data MSE + physics residual
    data = np.array(dataset, dtype=np.float32)
    X = torch.tensor(data[:, :6],  dtype=torch.float32).to(DEVICE) #features: T, S, eta1, eta2_eff, eta3, inp
    y = torch.tensor(data[:, 6:],  dtype=torch.float32).to(DEVICE) #targets: next_T, next_S

    net = StommelNet().to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) #adam optimizer
    loss_fn = nn.MSELoss()
    LAMBDA = 10.0 #physics weight — trust equations more than data

    net.train()
    EPOCHS = 200
    BATCH  = 512 #mini-batch size

    n = len(X)
    for epoch in range(EPOCHS):
        idx = torch.randperm(n, device=DEVICE) #shuffle each epoch
        for start in range(0, n, BATCH):
            batch_idx = idx[start:start + BATCH]
            X_batch = X[batch_idx]
            pred = net(X_batch)
            data_loss    = loss_fn(pred, y[batch_idx]) #how well model fits clean RK4 targets
            physics_loss = stommel_residual(X_batch, pred) #how much model violates Stommel ODEs
            loss = data_loss + LAMBDA * physics_loss #PINN total loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    net.eval()
    return net


def predict(net, features): #single inference — returns predicted next_T, next_S
    net.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).to(DEVICE)
        out = net(x).cpu().numpy()[0] #shape [2]
        return float(out[0]), float(out[1]) #next_T, next_S


def regimeAccuracy(net, dataset): #% of samples where predicted regime matches true regime
    data = np.array(dataset, dtype=np.float32)
    X = torch.tensor(data[:, :6], dtype=torch.float32).to(DEVICE)
    true_psi = data[:, 6] - data[:, 7] #next_T - next_S = true next psi
    with torch.no_grad():
        pred = net(X).cpu().numpy()
    pred_psi = pred[:, 0] - pred[:, 1] #predicted next_T - next_S
    return float(np.mean(np.sign(pred_psi) == np.sign(true_psi)) * 100)


# --- NODE INITIALIZATION ---
random.seed(42) #seed so positions are the same every restart
nodes = []
for i in range(NODE_COUNT):
    node = LearningNode(
        node_id=i+1,
        initial_T=random.uniform(1.0, 3.0),   #realistic stommel T range
        initial_S=random.uniform(0.3, 1.0),   #realistic stommel S range
        eta1=3.0,                              #thermal forcing, same for all per the study
        eta2=random.uniform(0.8, 1.2),         #salinity forcing, varies by location
        eta3=0.1,                              #diffusivity, same for all per the study
        x=random.uniform(0, 100),             #longitude-like position
        y=random.uniform(0, 100),             #latitude-like position
        z=random.uniform(0, 100)              #depth-like position
    )
    nodes.append(node)
random.seed() #unseed so runs stay random after init


# --- SNAPSHOT ---
def snapshot(): #log the current state of all nodes
    state_log.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"), #right now
        "states": {node.id: round(node.psi, 3) for node in nodes} #psi value for each node rounded to thousandth
    })


# --- BUILD 3D SCATTER DATA FOR PLOTLY ---
def build_graph_data(): #pack node positions and state into plotly-ready json
    th_nodes = [n for n in nodes if n.regime == "TH"] #split by regime for separate traces
    sa_nodes = [n for n in nodes if n.regime == "SA"]

    def trace(group, name, color): #helper to build one plotly trace dict
        return {
            "type": "scatter3d",
            "mode": "markers",
            "name": name,
            "x": [n.x for n in group],
            "y": [n.y for n in group],
            "z": [n.z for n in group],
            "text": [f"Node {n.id}<br>T={n.T:.3f}<br>S={n.S:.3f}<br>Ψ={n.psi:.3f}" for n in group], #hover text
            "hoverinfo": "text",
            "marker": {
                "size": 6,
                "color": [n.psi for n in group], #color by psi value
                "colorscale": [[0, color[0]], [1, color[1]]], #gradient within regime
                "showscale": False
            }
        }

    data = []
    if th_nodes: data.append(trace(th_nodes, "TH", ["#aed6f1", "#1a5276"])) #light to dark blue for TH
    if sa_nodes: data.append(trace(sa_nodes, "SA", ["#f1948a", "#922b21"])) #light to dark red for SA

    layout = {
        "paper_bgcolor": "#f5f0e8",
        "plot_bgcolor":  "#f5f0e8",
        "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
        "legend": {"x": 0, "y": 1},
        "scene": {
            "xaxis": {"title": "X", "backgroundcolor": "#f5f0e8"},
            "yaxis": {"title": "Y", "backgroundcolor": "#f5f0e8"},
            "zaxis": {"title": "Z (depth)", "backgroundcolor": "#f5f0e8"}
        }
    }
    return json.dumps({"data": data, "layout": layout})


# --- HOME ---
@app.route("/", methods=["GET"]) #home page
def home():
    global model

    if len(dataset) >= 5 and model is None: #train the model but not if we have under 5
        model = trainModel(dataset)

    display_nodes = nodes[:100] #cap table at 100 rows for readability

    html = """<html><head>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f5f0e8; color: #1a2e3b; margin: 20px; }
        h2 { color: #2e86c1; }
        table { border-collapse: collapse; width: 90%; margin-bottom: 20px; }
        th, td { border: 1px solid #aec6cf; padding: 8px; text-align: center; }
        th { background-color: #d6eaf8; color: #1a5276; }
        .section { margin-top: 24px; border-top: 2px solid #aec6cf; padding-top: 14px; }
        input[type=number] { width: 80px; padding: 4px; }
        input[type=submit] { background-color: #2e86c1; color: white; padding: 6px 14px; border: none; cursor: pointer; }
        input[type=submit]:hover { background-color: #1a5276; }
        button { background-color: #2e86c1; color: white; padding: 5px 12px; border: none; cursor: pointer; }
        button:hover { background-color: #1a5276; }
        a { color: #2e86c1; }
        p { margin: 4px 0; }
        .ni-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
        .ni { display: flex; flex-direction: column; font-size: 0.8rem; color: #5d8aa8; }
        .ni input { width: 80px; }
        #graph { width: 100%; height: 500px; }
    </style></head><body>
    """

    # 3d graph
    graph_json = build_graph_data()
    html += f"""<div class="section">
<h2>Node Space (3D)</h2>
<div id="graph"></div>
<script>
    var fig = {graph_json};
    Plotly.newPlot('graph', fig.data, fig.layout, {{responsive: true}});
</script>
</div>"""

    # node table — current state + predicted next Ψ + ΔΨ
    html += f"<div class='section'><h2>Nodes ({NODE_COUNT} fixed)</h2>"
    html += "<div style='max-height:400px;overflow-y:auto;width:90%'>"
    html += "<table><tr><th>ID</th><th>T</th><th>S</th><th>Ψ</th><th>Regime</th><th>η₁</th><th>η₂</th><th>Pred next T</th><th>Pred next S</th><th>Pred next Ψ</th><th>ΔΨ</th></tr>"
    for n in display_nodes:
        t = (n.psi - n.min_psi) / (n.max_psi - n.min_psi) #normalize psi to 0-1 for color
        t = max(0.0, min(1.0, t)) #clamp
        r = int(245 + (26  - 245) * t)
        g = int(240 + (86  - 240) * t)
        b = int(232 + (193 - 232) * t)
        cell_color = f"rgb({r},{g},{b})"
        regime_color = "#1a5276" if n.regime == "TH" else "#c0392b"
        if n.id in prediction_results:
            pred_T, pred_S = prediction_results[n.id]
            pred_psi = pred_T - pred_S #predicted next Ψ = pred_T - pred_S
            delta_psi = pred_psi - n.psi #how much model expects Ψ to change
            pred_psi_str  = f"{pred_psi:+.4f}"
            delta_psi_str = f"{delta_psi:+.4f}"
            delta_color = "#1a5276" if delta_psi >= 0 else "#c0392b" #blue strengthening, red weakening
        else:
            pred_T = pred_S = None
            pred_psi_str = delta_psi_str = "—"
            delta_color = "#5d8aa8"
        html += (f"<tr>"
                 f"<td>{n.id}</td>"
                 f"<td>{n.T:.4f}</td>"
                 f"<td>{n.S:.4f}</td>"
                 f"<td style='background:{cell_color};color:#fff;font-weight:600'>{n.psi:.4f}</td>"
                 f"<td style='color:{regime_color};font-weight:700'>{n.regime}</td>"
                 f"<td>{n.eta1:.2f}</td>"
                 f"<td>{n.eta2:.2f}</td>"
                 f"<td style='color:#2e86c1;font-weight:600'>{f'{pred_T:.4f}' if n.id in prediction_results else '—'}</td>"
                 f"<td style='color:#c0392b;font-weight:600'>{f'{pred_S:.4f}' if n.id in prediction_results else '—'}</td>"
                 f"<td style='color:#1a5276;font-weight:600'>{pred_psi_str}</td>"
                 f"<td style='color:{delta_color};font-weight:600'>{delta_psi_str}</td>"
                 f"</tr>")
    if NODE_COUNT > 100:
        html += f"<tr><td colspan='11' style='color:#5d8aa8;font-style:italic'>... and {NODE_COUNT-100} more</td></tr>"
    html += "</table></div></div>"

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

    # prediction — run model on explicit user inputs only
    html += "<div class='section'><h2>Predict next T and S</h2>"
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
        html += f"<p>StommelNet (PyTorch) &nbsp;|&nbsp; Target: next T, next S &nbsp;|&nbsp; Dataset rows: {len(dataset)}</p>"
        if len(dataset) >= 5:
            acc = regimeAccuracy(model, dataset)
            html += f"<p>Regime sign accuracy: {acc:.1f}%</p>"
            html += f"<p>dt = {DT} &nbsp;|&nbsp; noise σ = {NOISE_SIGMA} &nbsp;|&nbsp;</p>"
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


@app.route("/predict_inputs", methods=["POST"]) #run model on explicit user inputs
def predict_inputs():
    global model, prediction_results
    prediction_results = {}
    if model is None:
        return redirect(url_for('home'))
    for node in nodes:
        try: inp = float(request.form.get(f"n{node.id}", 0))
        except: inp = 0.0
        eta2_eff = node.eta2 + inp #effective salinity forcing with perturbation
        features = [[node.T, node.S, node.eta1, eta2_eff, node.eta3, inp]] #stommel features
        prediction_results[node.id] = predict(model, features) #returns (next_T, next_S)
    return redirect(url_for('home'))


# --- INPUT STEP ---
@app.route("/input_step", methods=["POST"])
def input_step():
    for node in nodes:
        try: inp = float(request.form.get(f"n{node.id}", 0))
        except: inp = 0.0
        old_T, old_S = node.T, node.S #capture before stepping
        eta2_eff = node.eta2 + inp #effective salinity forcing this step
        clean_next_T, clean_next_S = node.step(inputs=[inp], dt=DT, noise_sigma=NOISE_SIGMA, model=None)
        dataset.append([old_T, old_S, node.eta1, eta2_eff, node.eta3, inp, clean_next_T, clean_next_S]) #clean targets
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
            inp = random.uniform(-2.0, 2.0) #wider perturbation range for more diverse state coverage
            old_T, old_S = node.T, node.S #capture before stepping
            eta2_eff = node.eta2 + inp #effective salinity forcing this step
            clean_next_T, clean_next_S = node.step(inputs=[inp], dt=DT, noise_sigma=NOISE_SIGMA, model=None)
            dataset.append([old_T, old_S, node.eta1, eta2_eff, node.eta3, inp, clean_next_T, clean_next_S]) #clean targets
        snapshot()
    return redirect(url_for('home'))


@app.route("/train") #retrain
def train(): #define train
    global model
    if len(dataset) >= 5: #minimum 5 runs
        model = trainModel(dataset)
    return redirect(url_for('home'))


if __name__ == "__main__": #if main
    app.run(debug=True) #run