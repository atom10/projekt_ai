import yfinance as yf
from datetime import datetime

#Funkcja do pobierania i formatowania danych historycznych
def fetch_stock_data(ticker):
    # Zdefiniuj symbol giełdowy
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period='max')[['Close']]  # Pobierz tylko kolumnę 'Close'
    hist.reset_index(inplace=True)
    hist.columns = ['Date', 'Value']
    # Przekształć indeks (który jest w formacie datetime) na kolumnę
    hist['Date'] = hist['Date'].dt.strftime('%d-%m-%Y')
    return hist

#print(ManganeseStock)

def getValue(date, mineral):
    date = datetime.strptime(date, '%d-%m-%Y')
    # Sprawdzenie typu minerału i wybranie odpowiedniego DataFrame
    if mineral.lower() == 'manganese':
        ManganeseStock = fetch_stock_data('MN.V')
        df = ManganeseStock
    elif mineral.lower() == 'molybdenum':
        MolybdenumStock = fetch_stock_data('601958.SS')
        df = MolybdenumStock
    else:
        print("Mineral not found returning none")
        return None
    # Sprawdzenie, czy data istnieje w ramce danych
    if date.strftime('%d-%m-%Y') in df['Date'].values:
        # Jeśli data istnieje, zwróć odpowiadającą wartość
        return df.loc[df['Date'] == date.strftime('%d-%m-%Y'), 'Value'].iloc[0]
    else:
        # Jeśli data nie istnieje, znajdź najbliższy istniejący rekord
        nearest_date = min(df['Date'], key=lambda x: abs(date - datetime.strptime(x, '%d-%m-%Y')))
        return df.loc[df['Date'] == nearest_date, 'Value'].iloc[0] 

#print(getValue('01-01-2024', 'manganese'))
