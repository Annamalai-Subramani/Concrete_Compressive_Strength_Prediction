from flask import Flask,render_template,request
from src.concrete_strength_prediction.pipelines.prediction_pipeline import CustomData,PredictPipeline 
from src.concrete_strength_prediction.logger import logging

app = Flask(__name__)
app.static_folder="static"
app.template_folder="static/templates"

@app.route("/")
def Home():
    return render_template("home.html")

@app.route("/concrete_predict", methods=["GET", "POST"])
def ConcretePredict():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            cement=float(request.form.get("cement")),
            blast_furnace_slag=float(request.form.get("blast_furnace_slag")),
            fly_ash=float(request.form.get("fly_ash")),
            water=float(request.form.get("water")),
            superplasticizer=float(request.form.get("superplasticizer")),
            coarse_aggregate=float(request.form.get("coarse_aggregate")),
            fine_aggregate=float(request.form.get("fine_aggregate")),
            age=float(request.form.get("age"))
        )

        features = data.get_data_as_dataframe()
        logging.info(f'Feature data: {features.to_string()}')
        prediction = PredictPipeline().predict(features=features)

        return render_template("home.html", final_result=f'Predicted Compressive Strength: {prediction[0]:.2f} MPa')


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080, debug=True)