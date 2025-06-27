from flask import Flask
from route.predict_route import predict_blueprint

app = Flask(__name__)
app.register_blueprint(predict_blueprint)
for rule in app.url_map.iter_rules():
    print(f"{rule} -> methods={rule.methods}")

if __name__ == '__main__':
    app.run(debug=True)