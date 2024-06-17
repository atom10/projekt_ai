 # Instrukcja uruchomienia aplikacji w Dockerze na sytemie operacyjnym Windows
 ## 1. Pobranie repozytorium<br>
 Najpierw należy pobrać repozytorium z GitHub. W terminalu wykonaj poniższe polecenie:

        git clone https://github.com/atom10/projekt_ai.git


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