from flask import Flask, render_template, request, redirect, url_for, flash
import logging
import warnings
import os
import political_evaluation as pe
import weather_evaluation as we
import stock_evaluation as se
import logistic_evaluation as le
import llmAccess as llm
from datetime import datetime
import model
import data
import html_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.ERROR)

app = Flask(__name__)
app.secret_key = 'dkfleo13k4r3erdsfr24WER'

data_for_training = None
lstm_model = None
xgb_model = None
scaler = None

def validate_dates(start_date, end_date):
    """Validate date range input."""
    if not start_date or datetime.strptime(start_date, "%d-%m-%Y") < datetime.strptime("01-01-1990", "%d-%m-%Y"):
        return "Invalid start date."
    if not end_date or datetime.strptime(end_date, "%d-%m-%Y") > datetime.strptime("01-01-2025", "%d-%m-%Y"):
        return "Invalid end date."
    return None
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/action', methods=['POST'])
def action():
    global data_for_training, lstm_model, xgb_model, scaler

    action = request.form.get('action')
    if action == '1':
        prompt = "Who are you?"
        result = "Q: " + prompt + "\nA: " + llm.send_prompt(prompt)
    elif action == '2':
        result = str(pe.evaluate('manganese', '01-01-1970', '01-01-2025'))
    elif action == '3':
        result = str(we.weather_evaluation('manganese', '08-06-2024'))
    elif action == '4':
        result = str(se.getValue('01-01-2024', 'manganese'))
    elif action == '5':
        result = str(le.logistic_evaluation('01-01-2024'))
    elif action == '6':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        validation_error = validate_dates(start_date, end_date)
        if validation_error:
            result = validation_error
        else:
            data_for_training = data.generate_data(start_date, end_date)
            result = html_utils.generate_table_html(data_for_training, 20)
    elif action == '7':
        data.save_data(data_for_training)
        result = "Data saved successfully."
    elif action == '8':
        data_for_training = data.load_data()
        result = html_utils.generate_table_html(data_for_training,20)
    elif action == '9':
        lstm_model, xgb_model, scaler, mse, mae, mape = model.train_model(data_for_training)
        result = html_utils.generate_metric_paragraph('Mean Squared Error', mse)
        result += html_utils.generate_metric_paragraph('Mean Absolute Error', mae)
        result += html_utils.generate_metric_paragraph('Mean Absolute Percentage Error', mape)
        result += html_utils.generate_image('static/training_results.png')
    elif action == '10':
        model.save_models(lstm_model, xgb_model, scaler)
        result = "Models saved successfully."
    elif action == '11':
        lstm_model, xgb_model, scaler = model.load_models()
        result = "Models loaded successfully."
    elif action == '12':
        mineral = request.form.get('mineral')
        date = request.form.get('date')
        packet = data.generate_singe_data_packet(datetime.strptime(date, "%d-%m-%Y"), mineral, with_target=False)
        target = model.predict_target(packet, lstm_model, xgb_model, scaler)
        result = html_utils.generate_metric_paragraph(f"Predicted {mineral} price for {date}:", target[0])
    elif action == '13':
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        validation_error = validate_dates(start_date, end_date)
        if validation_error:
            result = validation_error
        else:
            lstm_model, xgb_model, scaler = model.load_models()
            additional_data = data.generate_data(start_date, end_date)
            lstm_model, xgb_model, scaler, mse, mae, mape = model.retrain_models(additional_data, lstm_model, xgb_model, scaler)
            result = html_utils.generate_metric_paragraph('Mean Squared Error', mse)
            result += html_utils.generate_metric_paragraph('Mean Absolute Error', mae)
            result += html_utils.generate_metric_paragraph('Mean Absolute Percentage Error', mape)
            result += html_utils.generate_image('static/training_results.png')
    else:
        result = "Invalid action."

    flash(result)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
