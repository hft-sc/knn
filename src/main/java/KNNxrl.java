public class KNNxrl implements KNN {
    // Feedforward-Neuronales Netz variabler Anzahl an Hiddenschichten

    /**
     * Describes the weight between to nodes
     * First dimension is layer
     * second dimension is input node index in layer
     * third dimension is output node index in layer + 1
     */
    public final double[][][] weight;

    /**
     * Network of all nodes, containing their node numbers
     * First dimension are the layers
     * Second dimension are the nodes in the layer
     * <p>
     * For each layer, except the last one, the first node is the bias
     */
    private final double[][] delta;
    /**
     * The input value for a given node
     * <p>
     * See {@link KNNxrl#delta} for dimensions
     */
    private final double[][] input;
    /**
     * The output value for a given node
     * <p>
     * See {@link KNNxrl#delta} for dimensions
     */
    private final double[][] output;
    private final int layerCount;
    private final double maxAlpha;
    private final double minAlpha;
    private final int maxEpoch;

    private double currentAlpha;

    public KNNxrl(int dimensions, TestParameters testParameters) {
        maxAlpha = testParameters.getMaxAlpha();
        minAlpha = testParameters.getMinAlpha();
        currentAlpha = maxAlpha;
        maxEpoch = testParameters.getMaxEpoch();

        final var layers = testParameters.getLayers();
        layerCount = layers.length + 2;// Anzahl Hiddenschichte + Eingabeschicht + Ausgabeschicht
        delta = new double[layerCount][];
        input = new double[layerCount][];
        output = new double[layerCount][];

        // Eingabeschicht
        // der erste Knoten der ersten Schicht ist ein Bias, deshalb plus 1
        delta[0] = new double[dimensions + 1];
        input[0] = new double[dimensions + 1];
        output[0] = new double[dimensions + 1];

        // Ausgabeschicht
        // es gibt einen Ausgabeknoten, da ein Klassifikationsproblem vorliegt
        delta[layerCount - 1] = new double[1];
        input[layerCount - 1] = new double[1];
        output[layerCount - 1] = new double[1];

        // Hiddenschichten
        for (int layer = 1; layer <= layers.length; layer++) {
            final var nodeInLayer = layers[layer - 1];
            delta[layer] = new double[nodeInLayer];
            input[layer] = new double[nodeInLayer];
            output[layer] = new double[nodeInLayer];
        }

        weight = new double[layerCount - 1][][];
        weight[0] = new double[dimensions + 1][];
        for (int inputNode = 0; inputNode < weight[0].length; inputNode++) {
            weight[0][inputNode] = new double[layers[0]];
        }

        for (int hiddenLayer = 0; hiddenLayer < layers.length; hiddenLayer++) {
            final var nodeCount = layers[hiddenLayer];
            weight[hiddenLayer + 1] = new double[nodeCount][];
            for (int inputNode = 0; inputNode < nodeCount; inputNode++) {
                var nextLayerNodeCount = layers[hiddenLayer];
                weight[hiddenLayer + 1][inputNode] = new double[nextLayerNodeCount];
            }
        }
    }

    @Override
    public void trainieren(double[][] liste, boolean print) {
        double[] fehlerVektor;

        double klasse;
        double fehler;
        int anzFehler;

        initWeights();

        int minAnzFehler = Integer.MAX_VALUE;
        double minFehler = Double.MAX_VALUE;
        boolean goBack = false;
        boolean stop = false;
        int epoche = 1;

        while (!stop) {
            epoche++;
            currentAlpha = currentAlpha - (maxAlpha - minAlpha) / maxEpoch;


            for (double[] doubles : liste) {
                initInputLayer(doubles);
                klasse = doubles[doubles.length - 1]; // 0 or 1. Because its a classification problem
                forward();
                backward(klasse);
            }

            fehlerVektor = fehler3(liste);
            fehler = fehlerVektor[0];
            anzFehler = (int) fehlerVektor[1];

            if (print) {
                System.out.println("-Epoche: " + epoche + " " + anzFehler + " " + fehler + " minAnzFehler " + minAnzFehler + " minFehler " + minFehler + " " + goBack + " " + maxAlpha);
            }
            if (epoche >= maxEpoch + 1 || anzFehler == 0) stop = true;
        }
    }

    /**
     * Forward-Pass
     */
    private void forward() {
        // Skip bias layer, so int layer = 1
        for (int layer = 1; layer < input.length; layer++) {
            // Skip bias node, so nodeNumber = 1
            for (int nodeNumber = 1; nodeNumber < input[layer].length; nodeNumber++) {
                input[layer][nodeNumber] = 0.0;
                final var previousLayer = layer - 1;
                for (int previousLayerNodeNumber = 0; previousLayerNodeNumber < input[previousLayer].length; previousLayerNodeNumber++) {
                    input[layer][nodeNumber] += weight[previousLayer][previousLayerNodeNumber][nodeNumber] * output[previousLayer][previousLayerNodeNumber];
                }
                output[layer][nodeNumber] = aktivierungsFunktion(input[layer][nodeNumber]);
            }
        }
    }

    /**
     * Calculates the backwards path and updates it directly
     *
     * @param klasse Whether it is a 0 or 1 in this classification
     */
    private void backward(double klasse) {
        deltaOutputLayer(klasse);
        deltaHiddenLayers();
        updateWeights();
    }

    /**
     * backward-Pass
     * <p>
     * delta = error
     */
    private void deltaOutputLayer(double klasse) {
        int outputLayer = layerCount - 1;
        for (int nodeNumber = 0; nodeNumber < output[outputLayer].length; nodeNumber++) {
            final var error = output[outputLayer][nodeNumber] - klasse;
            delta[outputLayer][nodeNumber] = ableitungAktivierungsFunktion(input[outputLayer][nodeNumber]) * error;
        }
    }

    private void deltaHiddenLayers() {
        int outputLayer = layerCount - 1;
        for (int layer = outputLayer - 1; layer >= 0; layer--) {
            for (int nodeNumber = 1; nodeNumber < delta[layer].length; nodeNumber++) {
                double sum = 0;
                for (int nodeNumberNextLayer = 0; nodeNumberNextLayer < delta[layer + 1].length; nodeNumberNextLayer++) {
                    sum += weight[layer][nodeNumber][nodeNumberNextLayer] * delta[layer + 1][nodeNumberNextLayer];
                }
                delta[layer][nodeNumber] = ableitungAktivierungsFunktion(input[layer][nodeNumber]) * sum;
            }
        }
    }

    private void updateWeights() {
        for (int layer = 0; layer < delta.length - 1; layer++) {
            for (int nodeNumber = 0; nodeNumber < delta[layer].length; nodeNumber++) {
                for (int nodeNumberNextLayer = 0; nodeNumberNextLayer < delta[layer + 1].length; nodeNumberNextLayer++) {
//                    final var gradient = output[layer][nodeNumber] * delta[layer + 1][nodeNumberNextLayer];
//                    final double delta = currentAlpha * gradient;

                    //Gradientenabstieg
                    weight[layer][nodeNumber][nodeNumberNextLayer] -= currentAlpha * output[layer][nodeNumber]
                            * delta[layer + 1][nodeNumberNextLayer];
                }
            }
        }
    }

    /**
     * Aktivierungsfunktion und deren Ableitung (sigmuid)
     */
    private double aktivierungsFunktion(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    private double ableitungAktivierungsFunktion(double x) {
        final var value = aktivierungsFunktion(x);
        return (value * (1 - value));
    }

    /**
     * Initialisierung
     */
    private void initWeights() {
        for (int layer = 0; layer < layerCount - 1; layer++) {
            for (int nodeNumber = 0; nodeNumber < delta[layer].length; nodeNumber++) {
                for (int nodeNumberNextLayer = 0; nodeNumberNextLayer < delta[layer + 1].length; nodeNumberNextLayer++) {
                    // random number between -1 and 1
                    weight[layer][nodeNumber][nodeNumberNextLayer] = Math.random() * 2 - 1;
                }
            }
        }
    }

    /**
     * Initialize all bias node
     *
     * @param input
     */
    private void initInputLayer(double[] input) {
        var bias = 0.1;
        // Init bias nodes. First node of each layer is bias
        for (int layer = 0; layer < this.input.length - 1; layer++) {
            this.input[layer][0] = bias;
            output[layer][0] = bias;
        }

        // Alle Knoten der Eingabeschicht ab dem 2. Knoten mit Eingabe belegen (1. Knoten ist ja Bias!)
        // der letzte Wert in input ist der output und gehört nicht zur Eingabe, deshalb input.length -1
        for (int nodeNumber = 0; nodeNumber < input.length - 1; nodeNumber++) {
            this.input[0][nodeNumber + 1] = input[nodeNumber];
            output[0][nodeNumber + 1] = input[nodeNumber];
        }
    }


    /**
     * @return An array with 2 values. First one is the sum of the difference of predicted and actual value.
     * Second one is the number of wrong predictions
     */
    private double[] fehler3(double[][] liste) {
        double[] fehler = {0.0, 0.0};

        double klasse;
        for (double[] doubles : liste) {
            initInputLayer(doubles);
            klasse = doubles[doubles.length - 1];
            forward();
            final var lastOutput = getLastOutputNodeInLastLayer();
            fehler[0] += Math.pow(klasse - lastOutput, 2);
            if ((lastOutput < 0.5 && (int) klasse == 1) || (lastOutput >= 0.5 && (int) klasse == 0)) {
                fehler[1]++;
            }
        }
        return fehler;
    }

    @Override
    public double[] evaluieren(double[][] liste) {
        //fuer Bankenbeispiel
        double output;
        int falschPositiv = 0;
        int falschNegativ = 0;
        int richtigPositiv = 0;
        int richtigNegativ = 0;
        int anzahlPositiv = 0;
        int anzahlNegativ = 0;

        double[] ergebnis = new double[12];

        for (int s = 0; s < liste.length; s++) {
            initInputLayer(liste[s]);
            output = liste[s][liste[s].length - 1];
            forward();
            final var lastOutput = getLastOutputNodeInLastLayer();
            if (lastOutput < 0.5 && (int) output == 1) {
                falschNegativ++;
                anzahlPositiv++;
            } else if (lastOutput >= 0.5 && (int) output == 1) {
                richtigPositiv++;
                anzahlPositiv++;
            } else if (lastOutput >= 0.5 && (int) output == 0) {
                falschPositiv++;
                anzahlNegativ++;
            } else if (lastOutput < 0.5 && (int) output == 0) {
                richtigNegativ++;
                anzahlNegativ++;
            } else {
                System.out.println("Error0 in Auswertung");
            }
        }
        if (anzahlPositiv != richtigPositiv + falschNegativ) System.out.println("Error1 in Auswertung");
        if (anzahlNegativ != richtigNegativ + falschPositiv) System.out.println("Error2 in Auswertung");
        if (anzahlPositiv + anzahlNegativ != liste.length) System.out.println("Error3 in Auswertung");


        ergebnis[0] = liste.length;
        ergebnis[1] = anzahlPositiv;
        ergebnis[2] = anzahlNegativ;
        ergebnis[3] = (double) anzahlPositiv / (double) liste.length;
        ergebnis[4] = (double) anzahlNegativ / (double) liste.length;
        ergebnis[5] = (double) (richtigPositiv + richtigNegativ) / (double) liste.length;
        ergebnis[6] = richtigPositiv;
        ergebnis[7] = falschPositiv;
        ergebnis[8] = richtigNegativ;
        ergebnis[9] = falschNegativ;
        ergebnis[10] = (double) richtigPositiv / (double) (richtigPositiv + falschNegativ);
        ergebnis[11] = (double) falschPositiv / (double) (richtigNegativ + falschPositiv);

        return ergebnis;
    }

    private double getLastOutputNodeInLastLayer() {
        final var lastOutputLayer = output[output.length - 1];
        return lastOutputLayer[lastOutputLayer.length - 1];
    }

}
