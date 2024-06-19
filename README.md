 # Instrukcja uruchomienia aplikacji w Dockerze na sytemie operacyjnym Windows
 ## 1. Pobranie repozytorium z GitHub (opcja A: klonowanie repozytorium)<br>
 Najpierw należy pobrać repozytorium z GitHub. W terminalu wykonaj poniższe polecenie:

        git clone https://github.com/atom10/projekt_ai.git


## 1. Pobranie repozytorium z GitHub (opcja B: pobranie archiwum ZIP)
Alternatywnie, możesz pobrać repozytorium jako archiwum ZIP. Otwórz przeglądarkę, przejdź do repozytorium na GitHubie 

        https://github.com/atom10/projekt_ai

, a następnie kliknij przycisk "Code" i wybierz opcję "Download ZIP". Rozpakuj archiwum w odpowiednim miejscu na swoim komputerze.


 ## 2. Uruchomienie Docker Desktop<br>
Upewnij się, że masz zainstalowany Docker Desktop i uruchom go.


## 3. Budowanie obrazów Docker<br>
 Przejdź do katalogu projektu, który został pobrany w kroku 1. W terminalu wykonaj poniższe polecenie:

        docker-compose build


## 4. Uruchomienie kontenerów<br>
Po zakończeniu budowania obrazów, uruchom kontenery za pomocą następującego polecenia:

        docker-compose up

## 5. Otwórz aplikację w przeglądarce<br>
Po uruchomieniu kontenerów, w terminalu pojawi się adres URL. Otwórz ten adres w przeglądarce internetowej, aby uzyskać dostęp do aplikacji.
Przykładowe adresy URL mogą wyglądać tak:

         http://127.0.0.1:5000
         http://172.18.0.3:5000



# Instrukcja obsługi aplikacji - najważniejsze funkcje

 ## 1. Wczytanie zapisanych w pliku modeli<br>
Gotowe, wytrenowane już model zapisane są w pliku, aby zacząć korzystać z aplikacji należy je wczytać. W tym celu należy kliknąć przycisk "Load Model from File".

## 2. Wyświetlenie wykresów przedstawiających jakość modelu<br>
Po wczytaniu modelu, można wyświetlić wykresy przedstawiające jakość modelu. W tym celu należy kliknąć przycisk "Show last training plot". 
Wykres po lewej stronie pokazuje stratę modelu podczas treningu i walidacji na przestrzeni epok, wskazując na proces uczenia się modelu. 
Wykres po prawej stronie przedstawia porównanie rzeczywistych wartości z wartościami przewidywanymi przez model, oceniając jego dokładność.

## 3. Porównanie rzeczywistych i przewidywanych cen
Po wczytaniu modelu, można porównać rzeczywiste ceny z cenami przewidywanymi przez model. W tym celu należy wprowadzić zakres dat w formacie "dd-mm-yyyy" oraz wybrać minerał z listy rozwijanej. Następnie kliknąć przycisk "compare true and predicted prices (works with dates in the past)".
Uwaga: ta opcja działa tylko dla cen z przeszłości, ponieważ nie jesteśmy w stanie pobrać chociażby danych politycznych z przyszłości, więc dla przyszłości nie zadziała.

## 4. Przewidywanie cen na kolejne dni
Aby przewidzieć ceny na kolejne dni, podaj liczbę dni oraz wybierz rodzaj surowca. Następnie kliknij przycisk "predict prices for next x days"

## 5. Przewidywanie cen minerału na jeden dzień
Aby przewidzieć cenę minerału na jeden dzień, podaj datę w formacie "dd-mm-yyyy" oraz wybierz rodzaj surowca. Następnie kliknij przycisk "Predict mineral price for single day"

## 6. Wczytanie danych z pliku
Umożliwia wczytanie i wyświetlenie danych zapisanych w pliku. W tym celu należy kliknąć przycisk "Load data from File".

## 7. Generowanie danych
Aplikacja umożliwia generowanie danych. W tym celu należy wprowadzić zakres dat w formacie "dd-mm-yyyy" oraz kliknąć przycisk "Data Generation".

## 8. Wytrenowanie nowych modeli
Aplikacja umożliwia wytrenowanie nowych modeli. W tym celu należy kliknąć przycisk "Train New Models". Wczęśniej jednak należy wczytać dane z pliku (krok 6) lub je wygenerować (krok 7).

## 9. Zapisanie modeli do pliku
Po wytrenowaniu nowych modeli, można je zapisać do pliku. W tym celu należy kliknąć przycisk "Save Model to File".

## 10. Dotrenowanie modeli
Po wczytaniu modeli z pliku, można je dotrenować. W tym celu należy kliknąć przycisk "Retrain Models". Wcześniej jednak należy wczytać dane z pliku (krok 6) lub je wygenerować (krok 7).