import political_evaluation as pe 
import weather_evaluation as we
import stock_evaluation as se
import llmAccess as llm

while True:
    print("Choose action:\n1. test LLM\n2. test political evaluation\n3. test weather evaluation\n4. test stock evaluation")
    action = input()
    if action=='1':
        print(llm.send_prompt("Who are you?"))
    elif action=='2':
        print(pe.evaluate('manganese','01-01-1970', '01-01-2025'))
    elif action=='3':
        print(we.weather_evaluation('manganese','08-06-2024'))
    elif action=='4':
        print(se.getValue('01-01-2024', 'manganese'))