import os
from flask import Flask, render_template, request, jsonify

base_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            template_folder=os.path.join(base_dir, '../templates'),
            static_folder=os.path.join(base_dir, '../static'))

robot_config = {
    "speed": 150,
    "status": "READY",
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/set_speed', methods=['POST'])
def set_speed():
    data = request.json
    new_speed = data.get('speed')

    robot_config['speed'] = int(new_speed)
    print(f"ปรับความเร็วเป็น: {robot_config['speed']}")

    return jsonify(success=True)

@app.route('/api/stop', methods=['POST'])
def stop_robot():
    robot_config['status'] = "STOPPED"
    print("EMERGENCY STOP ACTIVATED!")
    return jsonify(success=True)

def run_web_server():
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    run_web_server()