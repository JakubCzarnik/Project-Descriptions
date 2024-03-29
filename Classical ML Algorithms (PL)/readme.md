# Fundamenty **Machine Learning**

Celem projektu jest ukazanie mojej wiedzy dotyczącej algorytmów klasycznego uczenia maszynowego, implementując najpopularniejsze z nich od zera i wykorzystując na bazowych zbiorach danych. Postanowiłem wykorzystać do tego język wyższego poziomu **C++**, w tym bibliotekę **Eigen** oraz narzędzie **CMake**, które służy do automatycznego zarządzania procesem kompilacji wielu programów. Projekt był trudniejszy, ze względu na brak użycia gotowych bibliotek do przetwarzania danych, takich jak np. **NumPy** czy **Pandas** w **Pythonie**.

## Wczytywanie danych
Na początku zaimplementowałem klasę **DataFrame**, która posiada metody potrzebne do wczytywania danych z pliku **CSV**, przetwarzania danych (takie jak wyrzucanie kolumn, wierszy, pobieranie wycinku z danych zwane **slicingiem** czy standaryzacji kolumn potrzebnej do treningu) oraz metody do wizualizacji (takie jak head, wyświetlającej pierwsze wiersze danych, czy unique pozwalającej wyświetlić ilość unikalnych wartości w danej kolumnie). Oraz funkcje potrzebne do ewaluacji modeli, takiej jak mean squared error (**MSE**) czy Classification accuracy oraz funkcję do splitowania danych (DataFrame) na zbiór treningowy oraz walidacyjny.

## Algorytm **KMeans**
Jest to najpopularniejszy algorytm **unsupervised learning** do **klastrowania** danych. Algorytm inicjalizuje **centroidy** losowymi punktami z danych, a następnie iteracyjnie przypisuje każdy punkt do najbliższego centroidu i aktualizuje centroidy na podstawie przypisanych punktów.

[**Implementacja KMeans**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/kmean)

## Algorytm **k-Nearest Neighbors** (**KNN**)
Jest to jeden z prostszych algorytmów, ponieważ metoda fit polega po prostu na zapisaniu wartości zależnych i docelowych. Klasyfikuje dane na podstawie jego najbliższych sąsiadów w przestrzeni euklidesowej. Do optymalizowania obliczeń wykorzystałem strukturę danych **heap**.

[**Implementacja KNN**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/knn)

## Algorytm **Linear Regression** oraz **Logistic Regression**
Oba te algorytmy są bardzo do siebie podobne, różnica polega na tym że logistic regression transferuje przedział wyników regresji do dyskretnej przestrzeni (0, 1) tak aby móc binarnie sklasyfikować problem, korzystając z **thresholdu** (wyznaczonego progu, tutaj 0.5). Do treningu wykorzystałem funkcje straty MSE, iteracyjne obliczanie gradientu oraz algorytm Stochastic Gradient Descent (**SGD**).

[**Implementacja Linear Regression**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/linear_reggresion)

[**Implementacja Logistic Regression**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/logistic_reggresion)

## Algorytm **Naive Bayes**
Wykorzystuje **twierdzenie Bayesa** do klasyfikacji danych. Metoda fit jest odpowiedzialna za trening modelu, oblicza średnią, wariancję i prawdopodobieństwo a priori dla każdej klasy. Metoda predict jest odpowiedzialna za prognozowanie klas dla nowych danych, oblicza prawdopodobieństwa dla każdej klasy, a następnie wybiera klasę z najwyższym prawdopodobieństwem dla każdego przykładu.

[**Implementacja Naive Bayes**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/naive)

## Algorytm **Principal Component Analysis** (**PCA**)
Algorytm PCA jest techniką redukcji wymiarowości, która jest często używana do wizualizacji wysokowymiarowych danych oraz do przyspieszania algorytmów uczenia maszynowego. Metoda fit jest odpowiedzialna za trening modelu, centruje dane, oblicza macierz kowariancji, a następnie oblicza **wartości własne** i **wektory własne** macierzy kowariancji. Metoda transform jest odpowiedzialna za transformację danych do przestrzeni o mniejszej liczbie wymiarów.

[**Implementacja PCA**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp/tree/main/src/pca)

## Repozutorium Github 

[**Tutaj**](https://github.com/JakubCzarnik/Machine-Learning-in-Cpp) znajduję się implementacja całego projektu, wraz z jego opisami.

