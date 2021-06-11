public interface KNN {
    /**
     * Trainieren mit Backpropagation Algorithmus
     *
     * @param liste Muster
     */
    void trainieren(double[][] liste, boolean print);

    /**
     * Methoden zur Evaluierung
     */
    double[] evaluieren(double[][] liste);

    /**
     * für GUI_Beispiele
     * zeigt grafisch den output des NN fuer alle Punkte (x1, x2) im angegebenen Intervall
     * zeigt die angegenben Daten zum Vergleich ebenfalls an
     * die Ausgabe wurde von einem Studenten entwickelt!
     */
    void evaluierenGUIII(double[][] daten);

    int output(double[] x);

    /**
     * Methoden zur  Ausgabe der Netzparameter
     */
    void ausgabeBias();

    void ausgabeNetzStruktur();

    void ausgabeKnotenwerte();

    void ausgabeDelta();

    void ausgabeInputwerte();

    void ausgabeEingabeSchicht();

    void ausgabeW();
}
