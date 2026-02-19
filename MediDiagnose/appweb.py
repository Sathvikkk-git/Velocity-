from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    # Initial state when the page loads
    return render_template('index.html', prediction="Waiting...", risk_class="latent")

@app.route('/predict', methods=['POST'])
def predict():
    # Capture inputs from HTML form
    hb = float(request.form.get('hb', 0))
    mcv = float(request.form.get('mcv', 0))
    group = request.form.get('group')

    # Simple logic simulation (Replace with: model.predict([[hb, mcv...]]))
    if hb < 10:
        res, r_class = "CRITICAL", "critical"
    elif hb < 12:
        res, r_class = "LATENT", "latent"
    else:
        res, r_class = "LOW RISK", "low"

    return render_template('index.html', 
                           prediction=res, 
                           risk_class=r_class, 
                           hb_val=hb, 
                           group_val=group)

if __name__ == '__main__':
    app.run(debug=True)