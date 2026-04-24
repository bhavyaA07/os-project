import os
import subprocess
from flask import Flask, send_file, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    if not os.path.exists('dashboard_output.html'):
        return """
        <html>
            <body style='font-family: sans-serif; padding: 2rem;'>
                <h1>Dashboard Not Found</h1>
                <p>Please run the simulation first.</p>
                <form action='/run' method='POST'>
                    <button type='submit' style='padding: 10px 20px; font-size: 16px; cursor: pointer;'>Run Simulation</button>
                </form>
            </body>
        </html>
        """
    return send_file('dashboard_output.html')

@app.route('/run', methods=['POST'])
def run_simulation():
    try:
        print("[INFO] Running CPU Scheduling Simulation...")
        # Run the main.py script
        process = subprocess.run(['python', 'main.py'], capture_output=True, text=True, check=True)
        print("[INFO] Simulation completed.")
        return send_file('dashboard_output.html')
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": "Simulation failed.", "error": e.stderr}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("  CPU Scheduling RL Dashboard Server running at:")
    print("  http://localhost:5000/")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
