# Algorytm najszybszego spadku dla regresji wielomianowej

Projekt z przedmiotu **Matematyczne Fundamenty Informatyki (MFI)** poświęcony implementacji algorytmu **najszybszego spadku (Gradient Descent)** dla problemu regresji wielomianowej jednej zmiennej.

W projekcie zaimplementowano od podstaw wszystkie kluczowe elementy procesu uczenia modelu regresji wielomianowej, a następnie przeprowadzono eksperymenty na różnych zbiorach danych. Dodatkowo przygotowano interaktywny dashboard umożliwiający wizualizację wyników.

---

## Cel projektu

Dany jest zbiór obserwacji:

$$
(x_i, y_i), \quad i = 1, 2, \ldots, m
$$

gdzie:

- $x_i$ oznacza wartość cechy wejściowej,
- $y_i$ oznacza odpowiadającą jej wartość zmiennej objaśnianej,
- $m$ oznacza liczbę obserwacji w zbiorze danych.

Celem jest znalezienie wielomianu możliwie dobrze opisującego zależność pomiędzy zmiennymi (x) oraz (y).

Model regresji wielomianowej stopnia (n) ma postać:

$$
\hat{y} = \theta_0 + \theta_1x + \theta_2x^2 + \ldots + \theta_n x^n
$$

gdzie:

- $\hat{y}$ oznacza wartość przewidywaną przez model,
- $\theta_0, \theta_1, \ldots, \theta_n$ są parametrami modelu,
- $n$ jest stopniem wielomianu.

---

## Zawartość projektu

Projekt składa się z dwóch głównych części:

* **Notebook Jupyter** – zawiera pełne wyprowadzenie teoretyczne, implementację algorytmów oraz eksperymenty.
* **Dashboard Streamlit** – umożliwia interaktywną analizę wyników i porównywanie dopasowania modeli dla różnych zbiorów danych.

---

## Podstawy teoretyczne

### Normalizacja zmiennej wejściowej

W regresji wielomianowej kolejne potęgi zmiennej mogą przyjmować bardzo duże wartości, co prowadzi do problemów numerycznych oraz utrudnia działanie algorytmu optymalizacji.

Dlatego przed utworzeniem cech wielomianowych wykonywana jest standaryzacja:

$$
z_i = \frac{x_i - \mu}{\sigma}
$$

gdzie:

- $z_i$ to znormalizowana wartość obserwacji,
- $x_i$ to pierwotna wartość cechy,
- $\mu$ to średnia wartości $x$,
- $\sigma$ to odchylenie standardowe wartości $x$.

Normalizacja poprawia stabilność obliczeń oraz przyspiesza zbieżność algorytmu gradientowego.

---

### Konstrukcja cech wielomianowych

Dla wielomianu stopnia (n) tworzony jest wektor cech:

$$
\phi(x_i) =
\begin{bmatrix}
1 & x_i & x_i^2 & \ldots & x_i^n
\end{bmatrix}
$$

Dla wszystkich obserwacji otrzymujemy macierz cech:

$$
X =
\begin{bmatrix}
1 & x_1 & x_1^2 & \ldots & x_1^n \\
1 & x_2 & x_2^2 & \ldots & x_2^n \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_m & x_m^2 & \ldots & x_m^n
\end{bmatrix}
$$

Pierwsza kolumna składa się z jedynek i odpowiada wyrazowi wolnemu $\theta_0$.

Model można wtedy zapisać w postaci macierzowej:

$$
\hat{y}=X\theta
$$

gdzie:

- $X$ to macierz cech wielomianowych,
- $\theta$ to wektor parametrów modelu,
- $\hat{y}$ to wektor predykcji.

---

### Funkcja predykcji

Predykcja dla całego zbioru danych realizowana jest za pomocą mnożenia macierzy:

$$
\hat{y}=X\theta
$$

Takie podejście pozwala efektywnie obliczać przewidywania dla wszystkich obserwacji jednocześnie.

---

### Kwadratowa funkcja straty

Jako miarę jakości dopasowania wykorzystano funkcję średniego błędu kwadratowego:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

lub równoważnie:

$$
J(\theta) = \frac{1}{2m} \|X\theta - y\|^2
$$

gdzie:

- $J(\theta)$ oznacza wartość funkcji straty,
- $m$ oznacza liczbę obserwacji,
- $X\theta$ jest wektorem predykcji,
- $y$ jest wektorem wartości rzeczywistych.

Współczynnik $\frac{1}{2}$ upraszcza późniejsze wyprowadzenie gradientu.

---

### Gradient funkcji straty

Gradient funkcji straty względem parametrów modelu ma postać:

$$
\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)
$$

gdzie:

- $\nabla J(\theta)$ oznacza gradient funkcji straty,
- $X^T$ oznacza transpozycję macierzy cech,
- $X\theta - y$ jest wektorem błędów predykcji.

Gradient wskazuje kierunek najszybszego wzrostu funkcji celu. Aby minimalizować błąd, należy poruszać się w kierunku przeciwnym do gradientu.

---

### Algorytm najszybszego spadku

Minimalizacja funkcji straty realizowana jest metodą iteracyjną.

Aktualizacja parametrów wykonywana jest zgodnie ze wzorem:

$$
\theta^{(k+1)} = \theta^{(k)} - \alpha \nabla J(\theta^{(k)})
$$

gdzie:

- $\theta^{(k)}$ oznacza wektor parametrów w $k$-tej iteracji,
- $\theta^{(k+1)}$ oznacza wektor parametrów po aktualizacji,
- $\alpha$ jest współczynnikiem uczenia,
- $\nabla J(\theta^{(k)})$ jest gradientem funkcji straty w aktualnym punkcie.

Podczas uczenia zapisywana jest również historia wartości funkcji straty, co pozwala analizować proces zbieżności algorytmu.

Dodatkowo zastosowano kryterium zatrzymania oparte na zmianie wartości funkcji straty pomiędzy kolejnymi iteracjami.

---

## Eksperymenty

W notebooku przeprowadzono serię eksperymentów dla różnych zbiorów danych.

Dla każdego zbioru:

1. wykonywana jest normalizacja danych,
2. tworzone są cechy wielomianowe,
3. uruchamiany jest algorytm najszybszego spadku,
4. obliczana jest końcowa wartość funkcji straty,
5. zapisywane są wyniki dla różnych stopni wielomianu.

Badane są modele o stopniu od **1 do 10**.

Celem eksperymentów jest analiza wpływu stopnia wielomianu na jakość dopasowania modelu oraz identyfikacja najlepszego stopnia dla każdego zbioru danych.

---

## Wizualizacja wyników

Notebook zawiera między innymi:

* wykres zmian wartości funkcji straty podczas uczenia,
* wykres dopasowanego wielomianu do danych rzeczywistych,
* porównanie jakości dopasowania dla różnych stopni wielomianu,
* zestawienie najlepszych modeli dla poszczególnych zbiorów danych.

---

## Dashboard interaktywny

Projekt zawiera aplikację Streamlit umożliwiającą interaktywną analizę wyników.

Dashboard pozwala:

* wybrać zbiór danych,
* zmieniać stopień wielomianu,
* porównywać wartości rzeczywiste z wartościami przewidywanymi przez model,
* analizować dopasowanie modelu na interaktywnych wykresach Plotly.

Dzięki temu możliwe jest szybkie porównywanie zachowania modeli bez konieczności ponownego uruchamiania notebooka.

---

## Uruchomienie dashboardu

Instalacja wymaganych bibliotek:

```bash
pip install -r requirements.txt
```

Uruchomienie aplikacji:

```bash
streamlit run dashboard.py
```

Po uruchomieniu dashboard będzie dostępny pod adresem:

```text
http://localhost:8501
```

---

## Technologie

* Python
* NumPy
* Pandas
* Matplotlib
* Plotly
* Streamlit
* Jupyter Notebook

---

## Struktura projektu

```text
.
├── dashboard.py
├── polynomial_regression_gradient_descent.ipynb
├── data/
├── requirements.txt
└── README.md
```

Projekt stanowi praktyczną implementację regresji wielomianowej optymalizowanej metodą najszybszego spadku wraz z analizą wpływu stopnia wielomianu na jakość dopasowania oraz interaktywną prezentacją wyników.
