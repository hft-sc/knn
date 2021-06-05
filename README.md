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

Siehe `ExerciseA` zum austesten

