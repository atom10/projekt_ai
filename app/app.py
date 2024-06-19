from flask import Flask, render_template, request, redirect, url_for, flash
import logging
import warnings
import os
import political_evaluation as pe
import weather_evaluation as we
import stock_evaluation as se
import logistic_evaluation as le
import llmAccess as llm
from datetime import datetime, timedelta
import model
import data
import html_utils
import matplotlib.pyplot as plt

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

def track_state():
    model_status = 'present' if lstm_model is not None else 'missing'
    flash(model_status, 'model_status')
    data_status = 'present' if data_for_training is not None else 'missing'
    flash(data_status, 'data_status')

def validate_dates(start_date, end_date):
    """Validate date range input."""
    if not start_date or datetime.strptime(start_date, "%d-%m-%Y") < datetime.strptime("01-01-1990", "%d-%m-%Y"):
        return "Invalid start date."
    if not end_date or datetime.strptime(end_date, "%d-%m-%Y") > datetime.strptime("01-01-2025", "%d-%m-%Y"):
        return "Invalid end date."
    return None
@app.route('/')
def index():
    track_state()
    return render_template('index.html')
@app.route('/action', methods=['POST'])
def action():
    global data_for_training, lstm_model, xgb_model, scaler

    action = request.form.get('action')
    if action == '6':
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
        target = model.predict_target_from_single_date(packet, lstm_model, xgb_model, scaler)
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
    elif action == '14':
        result = html_utils.generate_image('static/training_results.png')
    elif action == '15':
        start_date = datetime.strptime(request.form.get('start_date'), "%d-%m-%Y")
        end_date = datetime.strptime(request.form.get('end_date'), "%d-%m-%Y")
        mineral = request.form.get('mineral')
        X_dates = []
        y_true = []
        y_predicted = []
        current_date = start_date
        while current_date <= end_date:
            X_dates.append(current_date)
            print(mineral)
            data_packet = data.generate_singe_data_packet(current_date, mineral, step=3, with_target=True)
            y_true.append(data_packet[-1])
            current_date += timedelta(days=3)
            y_predicted.append(model.predict_target_from_single_date(data_packet[:-1], lstm_model, xgb_model, scaler))
        plt.figure(figsize=(10, 6))
        plt.plot(X_dates, y_true, label='True Values', marker='o')
        plt.plot(X_dates, y_predicted, label='Predicted Values', marker='x')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title('True vs Predicted Values')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('static/performance_plot.png')
        plt.show()
        result = html_utils.generate_image('static/performance_plot.png')
    elif action == 'predict_next_x_days':
        days = int(request.form.get('days'))
        mineral = request.form.get('mineral')
        
        start_date = datetime.now()
        date_end = start_date + timedelta(days=days)
        
        date_list = data.generate_date_range(start_date.strftime("%d-%m-%Y"), date_end.strftime("%d-%m-%Y"))
        targets=[]
        for d in date_list:
            data_packet = data.generate_singe_data_packet(d, mineral, with_target=False)
            targets.append(model.predict_target_from_single_date(data_packet, lstm_model, xgb_model, scaler))
        dates = [date.strftime('%d-%m-%Y') for date in date_list]
        
        # result = "dates:" + str(date_list) + "<br> targets:" + str(targets)
        
        plt.figure(figsize=(10, 5))
        plt.plot(dates, targets, marker='o', linestyle='-')
        plt.xlabel('Date')
        plt.ylabel('Target')
        plt.title('Target Values over Time')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('static/predictions_for_n_days.png')
        plt.close()  # Close the plot to avoid displaying it in Flask
        result = html_utils.generate_image('static/predictions_for_n_days.png')
    else:
        result = "Invalid action."
    flash(result, 'result')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
