from datetime import datetime
from stock_evaluation import fetch_stock_data

logistic_stocks = []
for company in ['MATX', 'FDX', 'JBHT', '1919.HK']:
    logistic_stocks.append(fetch_stock_data(company))

def logistic_evaluation(date):
    date = datetime.strptime(date, '%d-%m-%Y')
    result=[]
    for companyData in logistic_stocks:
        # Sprawdzenie, czy data istnieje w ramce danych
        if date.strftime('%d-%m-%Y') in companyData['Date'].values:
            # Jeśli data istnieje, zwróć odpowiadającą wartość
            result.append(companyData.loc[companyData['Date'] == date.strftime('%d-%m-%Y'), 'Value'].iloc[0])
        else:
            # Jeśli data nie istnieje, znajdź najbliższy istniejący rekord
            nearest_date = min(companyData['Date'], key=lambda x: abs(date - datetime.strptime(x, '%d-%m-%Y')))
            result.append(companyData.loc[companyData['Date'] == nearest_date, 'Value'].iloc[0])
    return result

#print(logistic_evaluation('10-06-2024'))
