# application.py

"""
Flask web application for the student performance prediction model.

Routes:
- "/"           : Landing/index page.
- "/predictdata": Form page (GET) and prediction handler (POST).
"""

from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# `application` is often used by deployment platforms (e.g. gunicorn, AWS),
# while `app` is the usual local name.
application = Flask(__name__)
app = application


@app.route("/")
def index():
    """
    Render the index (landing) page.

    Returns:
        Rendered HTML template for the index page.
    """
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    """
    Handle the prediction form.

    GET:
        - Render the form page (home.html).

    POST:
        - Read form inputs.
        - Build a CustomData instance.
        - Convert inputs to a DataFrame.
        - Run the prediction pipeline.
        - Render the form page with the prediction result.

    Returns:
        Rendered HTML template for the form, optionally including
        the predicted result.
    """
    if request.method == "GET":
        return render_template("home.html")

    # POST: get data points from the submitted form
    try:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race/ethnicity"),
            parental_level_of_education=request.form.get(
                "parental level of education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test preparation course"),
            reading_score=float(request.form.get("reading score")),
            writing_score=float(request.form.get("writing score")),
        )

        # Convert inputs to a DataFrame matching training schema
        pred_df = data.get_data_as_frame()
        print(pred_df)  # Optional: helpful for debugging in the console

        # Run the prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Take the first prediction, convert to float, and round to 4 decimals
        predicted_score = float(results[0])
        predicted_score_rounded = round(predicted_score, 4)

        # results is typically a 1D array; we display the first prediction
        return render_template("home.html", results=predicted_score_rounded)

    except Exception as e:
        # In a real app you might want to show a user-friendly error page.
        # For now, just re-raise so it shows up in logs / console.
        raise e


if __name__ == "__main__":
    # Run the Flask development server
    app.run(host="0.0.0.0")
