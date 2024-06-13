import logging
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.ERROR)

import political_evaluation as pe 
import weather_evaluation as we
import stock_evaluation as se
import logistic_evaluation as le
import llmAccess as llm
from datetime import datetime, timedelta
import model
import data

data_for_training = None
lstm_model = None
xgb_model = None
scaler = None

while True:
    print(
        "Choose action:\n\
            1. test LLM\n\
            2. test political evaluation\n\
            3. test weather evaluation\n\
            4. test stock evaluation\n\
            5. test logistic evaluation\n\
            6. data generation\n\
            7. save data to file\n\
            8. load data from file\n\
            9. train new models\n\
            10. save model to file\n\
            11. load model drom file\n\
            12. predict mineral price\n\
            13. retrain\n\
            q. exit"
    )
    action = input()
    if action=='1':
        prompt = "Who are you?"
        print("Q: "+prompt) 
        print("A: "+llm.send_prompt(prompt))
    elif action=='2':
        print(pe.evaluate('manganese','01-01-1970', '01-01-2025'))
    elif action=='3':
        print(we.weather_evaluation('manganese','08-06-2024'))
    elif action=='4':
        print(se.getValue('01-01-2024', 'manganese'))
    elif action=='5':
        print(le.logistic_evaluation('01-01-2024'))
    elif action=='6':
        data_for_training = data.generate_data("01-01-2024", "01-02-2024")
        print(data_for_training)
    elif action=='7':
        data.save_data(data_for_training)
    elif action=='8':
        data_for_training = data.load_data()
        print(data_for_training)
    elif action=='9':
        lstm_model, xgb_model, scaler = model.train_model(data_for_training)
    elif action=='10':
        model.save_models(lstm_model, xgb_model, scaler)
    elif action=='11':
        lstm_model, xgb_model, scaler = model.load_models()
    elif action=='12':
        print("type one of: ")
        for mineral_name, _ in data.minerals.items():
            print(mineral_name + ", ")
        print("\n")
        mineral = input()
        print("date to predict in format dd-mm-yyyy")
        date = input()
        packet = data.generate_singe_data_packet(datetime.strptime(date, "%d-%m-%Y"), mineral, with_target=False)
        target = model.predict_target(packet, lstm_model, xgb_model, scaler)
        print(target)
    elif action=='13':
        print("start date for additional data in format dd-mm-yyyy: ")
        start_date = input()
        print("end date for additional data in format dd-mm-yyyy: ")
        end_date = input()
        lstm_model, xgb_model, scaler = model.load_models()
        additional_data = data.generate_data(start_date, end_date)
        lstm_model, xgb_model, scaler = model.retrain_models(additional_data, lstm_model, xgb_model, scaler)
    elif action=='q':
        pass
    