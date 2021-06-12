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
}
