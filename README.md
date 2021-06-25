# how to run

- extract the 2 zip files for a trimmed version of the training and test data
- or download and extract from https://www.kaggle.com/oddrationale/mnist-in-csv
- Adjust the file names in `ExerciseA.java` for `trainData` and `testData` if necessary

## Smaller dataset

- For faster runs trim the dataset with `head -n <number of rows> input.csv > output.csv`

# soft computing

## Aufgabe a

Auf Diabetes Datensatz bezogen

- Eine hohe Anzahl an Schichten verschlechtert das Ergebnis.  
  Bei einer/zwei Schichten, erhält man die besten Ergebnisse
- Ein hohe Knotenzahl verbessert das Ergebnis
- Ein hoher Alpha Wert verschlechtert das Ergebnis
- Eine hohe Epochen Anzahl verbessert das Ergebnis

So erhält man mit einer hohen Knotenanzahl, geringer Schichtenanzahl, geringem alpha und vielem Epochen eine sehr gutes
Ergebnis mit einer Genauigkeit von ~93%

```
             {layers=[8], alpha=0.5, maxEpoche=10}	Genauigkeit:0.7043	Trefferquote:0.1538	Ausfallrate:0.0132	richtigPositiv:   6	falschNegativ:  33	richtigNegativ:  75	falschPositiv:   1
 {layers=[8, 8, 8, 8, 8], alpha=0.5, maxEpoche=10}	Genauigkeit:0.6609	Trefferquote:0.0000	Ausfallrate:0.0000	richtigPositiv:   0	falschNegativ:  39	richtigNegativ:  76	falschPositiv:   0
             {layers=[8], alpha=1.0, maxEpoche=10}	Genauigkeit:0.6783	Trefferquote:0.0513	Ausfallrate:0.0000	richtigPositiv:   2	falschNegativ:  37	richtigNegativ:  76	falschPositiv:   0
           {layers=[8], alpha=100.0, maxEpoche=10}	Genauigkeit:0.3391	Trefferquote:1.0000	Ausfallrate:1.0000	richtigPositiv:  39	falschNegativ:   0	richtigNegativ:   0	falschPositiv:  76
            {layers=[8], alpha=1.0, maxEpoche=100}	Genauigkeit:0.8522	Trefferquote:0.7436	Ausfallrate:0.0921	richtigPositiv:  29	falschNegativ:  10	richtigNegativ:  69	falschPositiv:   7
          {layers=[8], alpha=1.0, maxEpoche=10000}	Genauigkeit:0.9391	Trefferquote:0.8974	Ausfallrate:0.0395	richtigPositiv:  35	falschNegativ:   4	richtigNegativ:  73	falschPositiv:   3
        {layers=[100], alpha=1.0, maxEpoche=10000}	Genauigkeit:0.9304	Trefferquote:0.8718	Ausfallrate:0.0395	richtigPositiv:  34	falschNegativ:   5	richtigNegativ:  73	falschPositiv:   3
```

Siehe `ExerciseA` in der `exerciseA` getaggten version zum austesten

## Aufgabe b

Hier wird das alpha dynamisch angepasst. Man kann in den KNN ein `maxAlpha` und ein `minAlpha` übergeben. Zu Beginn
wird `maxAlpha` benutzt und in jeder Epoche nähert man sich näher an `minAlpha` an.  
Der Wert nähert sich komplett linear von `maxAlpha` an `minAlpha` an

````java
double maxAlpha = 10.0;
double minAlpha = 1.0;
double currentAlpha = maxAlpha;
for(int i = 0; i < maxEpoche; i++) {
    currentAlpha-=(maxAlpha-minAlpha)/(maxEpoche);
}
````

An diesen Testdaten, getestet mit dem Diabetes Datensatz, sieht man dass man hiermit viel schneller auf bessere
Ergebnisse kommt. Vergleicht man den ersten Durchlauf mit den Dritten, sieht man eine deutlische Verbesserung der
Genauigkeit, Trefferquote und Ausfallrate. Um mit konstanten alpha ein ähnliches Ergebnis zu erreichen, benötigt man
10mal so viele Epochen.

Hier gilt aber weiterhin, dass ein zu großes Alpha das Ergebnis verschlechtert.

```
{layers=[8], maxAlpha=1.0, minAlpha=1.0, maxEpoche=1000}	Genauigkeit:0.7478	Trefferquote:0.4615	Ausfallrate:0.1053	richtigPositiv:  18	falschNegativ:  21	richtigNegativ:  68	falschPositiv:   8
{layers=[8], maxAlpha=1.0, minAlpha=1.0, maxEpoche=10000}	Genauigkeit:0.9304	Trefferquote:0.8718	Ausfallrate:0.0395	richtigPositiv:  34	falschNegativ:   5	richtigNegativ:  73	falschPositiv:   3
{layers=[8], maxAlpha=10.0, minAlpha=1.0, maxEpoche=1000}	Genauigkeit:0.9391	Trefferquote:0.8974	Ausfallrate:0.0395	richtigPositiv:  35	falschNegativ:   4	richtigNegativ:  73	falschPositiv:   3
{layers=[8], maxAlpha=50.0, minAlpha=1.0, maxEpoche=1000}	Genauigkeit:0.6609	Trefferquote:0.0000	Ausfallrate:0.0000	richtigPositiv:   0	falschNegativ:  39	richtigNegativ:  76	falschPositiv:   0
```
