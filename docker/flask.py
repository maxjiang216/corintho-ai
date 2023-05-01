from flask import Flask, request, jsonify
from random_move import get_random_move

app = Flask(__name__)

@app.route('/choose_move', methods=['POST'])
def choose_move():
    data = request.get_json()
    print(data)
    return jsonify(get_random_move(data['gameState'], data['timeLimit']))

if __name__ == '__main__':
    app.run(debug=True)