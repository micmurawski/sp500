# Temat: Wycena spółek i selekcja zwycięzców i przegranych na podstawie wyników finansowych.

## Pozyskiwanie danych

W celu zobycia danych z wyników finansowych spółek zamierzam zeskrobać dane ze strony
[macrotrends.net](https://www.macrotrends.net). W swojej analizie skupię się na spółkach z indeksu SP500. Spółki należące obecnie do indeksu pobiorę ze publiczego repozytorium github pod [linkiem](https://github.com/datasets/s-and-p-500-companies/blob/main/data/constituents.csv). Dodatkowo będę korzystał z [yfinance](https://github.com/ranaroussi/yfinance) w celu uzyskania historycznych wycen akcji.

## Cel badania

Celem badania będzie stworzenie dwóch modeli na podstawie danych z tzw. `balance sheet`, `income statements`, `cash flow statement`, `financial ratios`.

### Model wyceny spółek
model ten na podstawie danych finansowych będzie startał się przewidzieć. Cenę spółki, na podstawie danych finansowych. Zapewne będzie to model regesji liniowej.

$f(BS_i, IS_i, CF_i) = p_{i} $

$f$ - model

$p$ - balance sheet w okresie i (np. Q1 2022)

$BS_i$ - balance sheet w okresie i (np. Q1 2022)

$IS_i$ - income statement w okresie i (np. Q1 2022)

$CF_i$ - cash flow statement w okresie i (np. Q1 2022)

### Model selekcji wygranych przegranych

model ten będzie robił selekcję spółek na wygranych i przegranych czyli takie które w danym okresie czasu zyskały na wartości, lub zyskały na wartości powyej
indeksu S&P500. 

W tym przypadku zakładam następującą hipotezę zakładam hipotezę: 
Jesteśmy w stanie przewidzieć zwroty spóki $ r_{i+1} $ mając do dyspozycji
dane finansowe z poprzedniego okresu $BS_i, IS_i, CF_i$


## Złożoność zbioru
### Dane finansowe [link](https://github.com/micmurawski/sp500/blob/main/sp500/data.csv)

- Zbiór danych 25610 linii rekordów danych finansowych firm z ineksu S&P500 (data raportu kwartalnego x spółka)
- 421 pomyślnie pobranych danych spółek

### Dane wyceny [link](https://github.com/micmurawski/sp500/blob/main/sp500/target.csv)
- 499 pomyślnie pobranych wycen spółek i 421 wspólnych tickerów spółek z danymi finansowymi

## Metody analizy danych

* Wycena:

    * regresja liniowa - [OLS](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS)

* Selekcja zwycięzców/przegranych
    * Drzewa Losowe
    * KNN
    * Regresja Logistyczna

Dodatkowo rozważam zastosowanie analizy szeregów czasowych mając dane kwartalen wyników.

## Repozytorium GH
[https://github.com/micmurawski/sp500](https://github.com/micmurawski/sp500)