
public class KNNxrl implements KNN {
    // Feedforward-Neuronales Netz variabler Anzahl an Hiddenschichten

    private final int layerCount;
    private final int nodeCount;
    private final double[] input;
    private final double[] output;
    private final double[] delta;
    private final double maxAlpha;
    private final double minAlpha;
    private final int maxEpoch;
    /**
     * Network of all nodes, containing their node numbers
     * First dimension are the layers
     * Second dimension are the nodes in the layer
     */
    public int[][] network;
    public double[][] weight;
    /**
     * Whether a numbered node is a bias
     */
    public boolean[] bias; // true: Knoten ist Bias
    private double currentAlpha;

    public KNNxrl(int dataSize, TestParameters testParameters) {
        this.maxAlpha = testParameters.getMaxAlpha();
        this.minAlpha = testParameters.getMinAlpha();
        this.currentAlpha = maxAlpha;
        this.maxEpoch = testParameters.getMaxEpoche();

        this.layerCount = testParameters.getLayers().length + 2;// Anzahl Hiddenschichte + Eingabeschicht + Ausgabeschicht
        network = new int[layerCount][];
        int knotenNr = 0;

        // Eingabeschicht
        // der erste Knoten der ersten Schicht ist ein Bias, deshalb plus 1
        network[0] = new int[dataSize + 1];

        // Ausgabeschicht
        // es gibt einen Ausgabeknoten, da ein Klassifikationsproblem vorliegt
        network[layerCount - 1] = new int[1];

        // Hiddenschichten
        for (int l = 0; l < testParameters.getLayers().length; l++) {
            network[l + 1] = new int[testParameters.getLayers()[l]];
        }

        // alle Schichten werden mit fortlaufenden Knotennummern gefüllt
        for (int l = 0; l < layerCount; l++) {
            for (int i = 0; i < network[l].length; i++) {
                network[l][i] = knotenNr;
                knotenNr++;
            }
        }

        this.nodeCount = knotenNr;
        this.weight = new double[this.nodeCount][this.nodeCount];
        this.bias = new boolean[this.nodeCount];
        this.input = new double[this.nodeCount];
        this.output = new double[this.nodeCount];
        this.delta = new double[this.nodeCount];

        for (int l = 0; l < layerCount; l++) {
            for (int i = 0; i < network[l].length; i++) {
                knotenNr = network[l][i];
                // der erste Knoten einer Schicht wird bias (aussnahme in der ausgabeschicht)
                bias[knotenNr] = i == 0 && l < layerCount - 1;
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
                eingabeSchichtInitialisieren(doubles);
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
        for (int layer = 1; layer < network.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < network[layer].length; nodeNumber++) {
                int nodeNumberInCurrentLayer = network[layer][nodeNumber];
                if (!bias[nodeNumberInCurrentLayer]) {
                    input[nodeNumberInCurrentLayer] = 0.0;
                    for (int nri = 0; nri < network[layer - 1].length; nri++) {
                        int nodeNumberInPreviousLayer = network[layer - 1][nri];
                        input[nodeNumberInCurrentLayer] += weight[nodeNumberInPreviousLayer][nodeNumberInCurrentLayer] * output[nodeNumberInPreviousLayer];
                    }
                    output[nodeNumberInCurrentLayer] = aktivierungsFunktion(input[nodeNumberInCurrentLayer]);
                }
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
        int outputLayer = network.length - 1;
        for (int nrj = 0; nrj < network[outputLayer].length; nrj++) {
            int outputNodeNumber = network[outputLayer][nrj];
            final var error = output[outputNodeNumber] - klasse;
            delta[outputNodeNumber] = ableitungAktivierungsFunktion(input[outputNodeNumber]) * error;
        }
    }

    private void deltaHiddenLayers() {
        int ausgabeSchicht = network.length - 1;

        for (int l = ausgabeSchicht - 1; l >= 0; l--) {
            for (int nri = 0; nri < network[l].length; nri++) {
                int i = network[l][nri];
                delta[i] = 0.0;
                if (!bias[i]) {
                    double sum = 0;
                    for (int nrj = 0; nrj < network[l + 1].length; nrj++) {
                        int j = network[l + 1][nrj];
                        sum += weight[i][j] * delta[j];
                    }
                    delta[i] = ableitungAktivierungsFunktion(input[i]) * sum;
                }
            }
        }
    }

    private void updateWeights() {
        for (int l = 0; l < network.length - 1; l++) {
            for (int nri = 0; nri < network[l].length; nri++) {
                int i = network[l][nri];

                for (int nrj = 0; nrj < network[l + 1].length; nrj++) {
                    int j = network[l + 1][nrj];
                    if (!bias[j]) {
                        final var gradient = output[i] * delta[j];
                        double delt = currentAlpha * gradient;
                        weight[i][j] -= delt;//Gradientenabstieg
                    }
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
        return (aktivierungsFunktion(x) * (1 - aktivierungsFunktion(x)));
    }

    /**
     * Initialisierung
     */
    private void initWeights() {
        for (int layer = 0; layer < layerCount - 1; layer++) {
            for (int startNodeIndex = 0; startNodeIndex < network[layer].length; startNodeIndex++) {
                int startNode = network[layer][startNodeIndex];
                for (int j = 0; j < network[layer + 1].length; j++) {
                    int endNode = network[layer + 1][j];
                    if (!bias[endNode]) {
                        weight[startNode][endNode] = Math.random();
                        if (Math.random() < 0.5)
                            weight[startNode][endNode] = -weight[startNode][endNode];
                    }
                }
            }
        }
    }

    /**
     * Initialize all bias node
     *
     * @param input
     */
    private void eingabeSchichtInitialisieren(double[] input) {
        // Alle Bias-Knoten initialisieren
        for (int i = 0; i < network.length - 1; i++) {// über alle Schichten
            int knoten = network[i][0]; // der erste Knoten einer Schicht ist Bias!
            if (!bias[knoten])
                System.out.println("ups, nicht-Bias-Knoten als bias initialisiert");
            this.input[knoten] = 1.0;
            output[knoten] = 1.0;
        }

        // Alle Knoten der Eingabeschicht ab dem 2. Knoten mit Eingabe belegen (1. Knoten ist ja Bias!)
        // der letzte Wert in input ist der output und gehört nicht zur Eingabe, deshalb input.length -1
        for (int i = 0; i < input.length - 1; i++) {
            // in[0] ist Bias, deshalb i.ten Input bei bei in[i+1] speichern
            this.input[i + 1] = input[i];
            output[i + 1] = input[i];
        }
    }

    /**
     * Hilfsmethoden zur Evaluation
     */
    private int fehler(double[][] liste) {
        int anzFehler = 0;
        double klasse;
        for (int s = 0; s < liste.length; s++) {
            eingabeSchichtInitialisieren(liste[s]);
            klasse = liste[s][liste[s].length - 1];
            forward();
            if ((output[nodeCount - 1] < 0.5 && (int) klasse == 1) || (output[nodeCount - 1] >= 0.5 && (int) klasse == 0))
                anzFehler++;
            //System.out.println(s + " " + liste[s][0] + " " + liste[s][1] + " " + liste[s][2] + " " + a[n-1]);
        }
        return anzFehler;
    }

    private double fehler2(double[][] liste) {
        double fehler = 0.;
        double klasse;
        for (int s = 0; s < liste.length; s++) {
            eingabeSchichtInitialisieren(liste[s]);
            klasse = liste[s][liste[s].length - 1];
            forward();
            fehler += Math.pow(klasse - output[nodeCount - 1], 2);
        }
        return fehler;
    }

    /**
     * @return An array with 2 values. First one is the sum of the difference of predicted and actual value.
     * Second one is the number of wrong predictions
     */
    private double[] fehler3(double[][] liste) {
        double[] fehler = {0.0, 0.0};

        double klasse;
        for (double[] doubles : liste) {
            eingabeSchichtInitialisieren(doubles);
            klasse = doubles[doubles.length - 1];
            forward();
            fehler[0] += Math.pow(klasse - output[nodeCount - 1], 2);
            if ((output[nodeCount - 1] < 0.5 && (int) klasse == 1) || (output[nodeCount - 1] >= 0.5 && (int) klasse == 0)) {
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
            eingabeSchichtInitialisieren(liste[s]);
            output = liste[s][liste[s].length - 1];
            forward();
            if (this.output[nodeCount - 1] < 0.5 && (int) output == 1) {
                falschNegativ++;
                anzahlPositiv++;
            } else if (this.output[nodeCount - 1] >= 0.5 && (int) output == 1) {
                richtigPositiv++;
                anzahlPositiv++;
            } else if (this.output[nodeCount - 1] >= 0.5 && (int) output == 0) {
                falschPositiv++;
                anzahlNegativ++;
            } else if (this.output[nodeCount - 1] < 0.5 && (int) output == 0) {
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

    @Override
    public void evaluierenGUIII(double[][] daten) {
        int z = 0;
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                for (int k = 0; k < 10; k++) {
                    double wert1 = (double) i + (double) k / 10;
                    double wert2 = (double) j + (double) k / 10;
                    double[] werte = {wert1, wert2};
                    int erg = output(werte);
                    if (erg == 0) z++;
                }
            }
        }
        double[][] GUIwerteFlaeche = new double[z][2];
        z = 0;
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                for (int k = 0; k < 10; k++) {
                    double wert1 = (double) i + (double) k / 10;
                    double wert2 = (double) j + (double) k / 10;
                    double[] werte = {wert1, wert2};
                    int erg = output(werte);
                    if (erg == 0) {
                        GUIwerteFlaeche[z][0] = wert1;
                        GUIwerteFlaeche[z][1] = wert2;
                        z++;
                    }
                }
            }
        }
        GUI.zeichnen(daten, GUIwerteFlaeche);
    }

    @Override
    public int output(double[] x) {
        double[] input = new double[3];
        input[0] = x[0] / 100.;
        input[1] = x[1] / 100.;
        input[2] = 1.;//wird nicht benötigt

        eingabeSchichtInitialisieren(input);
        forward();

        double u = output[nodeCount - 1];
        int out;

        if (u < 0.5) out = 0;
        else out = +1;

        return out;
    }

    @Override
    public void ausgabeBias() {
        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < network.length; l++) {
                if (i < network[l].length) {
                    int k = network[l][i];
                    System.out.print(bias[k] + "\t");
                    ende = false;
                } else {
                    System.out.print("\t");
                }
            }
            System.out.println();
            i++;
        }
    }

    @Override
    public void ausgabeNetzStruktur() {
        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < network.length; l++) {
                if (i < network[l].length) {
                    System.out.print(network[l][i] + "\t\t");
                    ende = false;
                } else {
                    System.out.print("\t\t");
                }
            }
            System.out.println();
            i++;
        }
    }

    @Override
    public void ausgabeKnotenwerte() {
        System.out.println("Ausgabe der Knoten-Ausgabewerte");

        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < network.length; l++) {
                if (i < network[l].length) {
                    int k = network[l][i];
                    double preci = 1000;
                    double round = ((int) (output[k] * preci)) / preci;
                    System.out.print(round + "\t");
                    ende = false;
                } else {
                    System.out.print("\t");
                }
            }
            System.out.println();
            i++;
        }
    }

    @Override
    public void ausgabeDelta() {
        System.out.println("Ausgabe der DELTA");
        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < network.length; l++) {
                if (i < network[l].length) {
                    int k = network[l][i];
                    double preci = 100000;
                    double round = ((int) (delta[k] * preci)) / preci;
                    System.out.print(round + "\t");
                    ende = false;
                } else {
                    System.out.print("\t");
                }
            }
            System.out.println();
            i++;
        }
    }

    @Override
    public void ausgabeInputwerte() {
        System.out.println("Ausgabe der Inputwerte");
        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < network.length; l++) {
                if (i < network[l].length) {
                    int k = network[l][i];
                    double preci = 1000;
                    double round = ((int) (input[k] * preci)) / preci;
                    System.out.print(round + "\t");
                    ende = false;
                } else {
                    System.out.print("\t");
                }
            }
            System.out.println();
            i++;
        }
    }

    @Override
    public void ausgabeEingabeSchicht() {
        System.out.println("Ausgabe der Eingabeschicht");

        for (int i = 0; i < network[0].length; i++) {
            int knoten = network[0][i];
            System.out.print(output[knoten] + " ");
        }
        System.out.println();
    }

    @Override
    public void ausgabeW() {
        System.out.println("Ausgabe der Gewichte");

        for (int i = 0; i < nodeCount; i++) {
            for (int j = 0; j < nodeCount; j++) {
                double preci = 1000;
                double round = ((int) (weight[i][j] * preci)) / preci;
                if (round != 0)
                    System.out.println(i + " " + j + " " + round);
            }
            // System.out.println();
        }
        System.out.println();
    }

}
