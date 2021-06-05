public class KNN {
    // Feedforward-Neuronales Netz variabler Anzahl an Hiddenschichten

    public int[][] netz;        // Enthaelt pro Schicht netz[i] die enthaltenen Knotennummern
    public double[][] w;        // Gewichte
    // Knoteninformationen, jeweils durch ein Array gespeichert, Index =
    // Knotennummer
    public boolean[] bias; // true: Knoten ist Bias
    private int m;                // Anzahl Schichten
    private int n;                // Anzahl Knoten (insgesamt ueber alle Schichten)
    private double[] in;
    private double[] a;
    private double[] delta;

    private int batchSize = 1000;
    private double[][] gradient;

    /**
     * Fehlerrate für Backprobagation. Gewichtung des Gradienten
     */
    private double alpha;
    /**
     * Anzahl Iterationen bei Fehlerminimierung
     */
    private int maxIter = 1;
    /**
     * Anzahl Iterationen bei Fehlerminimierung
     */
    private int maxEpoche;


    /**
     * @param anzahlEingabewerte
     * @param anzahlKnotenProHiddenSchicht The length of array describes number of hidden layers.
     *                                     Each value describes how many nodes are in the respective layer
     */
    public KNN(int anzahlEingabewerte, int[] anzahlKnotenProHiddenSchicht, double alpha, int maxEpoche) {
        this.alpha = alpha;
        this.maxEpoche = maxEpoche;

        this.m = anzahlKnotenProHiddenSchicht.length + 2;// Anzahl Hiddenschichte + Eingabeschicht + Ausgabeschicht
        netz = new int[m][];
        int knotenNr = 0;

        // Eingabeschicht
        // der erste Knoten der ersten Schicht ist ein Bias, deshalb plus 1
        netz[0] = new int[anzahlEingabewerte + 1];

        // Ausgabeschicht
        // es gibt einen Ausgabeknoten, da ein Klassifikationsproblem vorliegt
        netz[m - 1] = new int[1];

        // Hiddenschichten
        for (int l = 0; l < anzahlKnotenProHiddenSchicht.length; l++) {
            netz[l + 1] = new int[anzahlKnotenProHiddenSchicht[l]];
        }

        // alle Schichten werden mit fortlaufenden Knotennummern gefüllt
        for (int l = 0; l < m; l++) {
            for (int i = 0; i < netz[l].length; i++) {
                netz[l][i] = knotenNr;
                knotenNr++;
            }
        }

        this.n = knotenNr;
        this.w = new double[this.n][this.n];
        this.bias = new boolean[this.n];
        this.in = new double[this.n];
        this.a = new double[this.n];
        this.delta = new double[this.n];
        this.gradient = new double[this.n][this.n];

        for (int l = 0; l < m; l++) {
            for (int i = 0; i < netz[l].length; i++) {
                knotenNr = netz[l][i];
                // der erste Knoten einer Schicht wird bias (aussnahme in der ausgabeschicht)
                bias[knotenNr] = i == 0 && l < m - 1;
            }
        }
    }

    /**
     * Trainieren mit Backpropagation Algorithmus
     *
     * @param liste Muster
     */
    public void trainieren(double[][] liste) {
        double[] fehlerVektor;

        double klasse;
        double fehler;
        int anzFehler;

        gewichteInitialisieren();

        int minAnzFehler = Integer.MAX_VALUE;
        double minFehler = Double.MAX_VALUE;
        boolean goBack = false;
        boolean stop = false;
        int epoche = 0;

        while (!stop) {
            epoche++;

            for (double[] doubles : liste) {
                eingabeSchichtInitialisieren(doubles);
                klasse = doubles[doubles.length - 1]; // 0 or 1. Because its a classification problem
                forward();
                backward(klasse);
            }

            fehlerVektor = fehler3(liste);
            fehler = fehlerVektor[0];
            anzFehler = (int) fehlerVektor[1];

            System.out.println("-Epoche: " + epoche + " " + anzFehler + " " + fehler + " minAnzFehler " + minAnzFehler + " minFehler " + minFehler + " " + goBack + " " + alpha);
            if (epoche >= maxEpoche || anzFehler == 0) stop = true;

//			if(anzFehler < minAnzFehler || (anzFehler == minAnzFehler && fehler < minFehler)) {//neue Bestlösung
//				minAnzFehler = anzFehler;
//				for(int i=0;i<n;i++){
//					for(int j=0;j<n;j++){
//						optAnzGewichte[i][j] = w[i][j];
//					}
//				}
//		    }

//			if(fehler < minFehler && anzFehler == minAnzFehler) {//neue Bestlösung
//				minFehler = fehler;
//				for(int i=0;i<n;i++){
//					for(int j=0;j<n;j++){
//						optFehGewichte[i][j] = w[i][j];
//					}
//				}
//				System.out.println("-----------------------------------------------------> neuer Minfehler " + anzFehler + " " + fehler);
//				anzVb++;
//		    }

/*			
			
//			if(epoche%intervall==0){
			if(anzFehler > minAnzFehler && minAnzFehler <10){//TOP!!!!!!!!!!!!
		//	if(anzFehler > minAnzFehler +1){//TOP!!!!!!!!!!!!

//			if(anzFehler > minAnzFehler ) {
				for(int i=0;i<n;i++){
					for(int j=0;j<n;j++){
						w[i][j] = optFehGewichte[i][j];/////
					}
				}
				for (int s = 0; s < liste.length; s++) {
					eingabeSchichtInitialisieren(liste[s]);
					klasse = liste[s][liste[s].length - 1];
					forward();
				}
				fehlerVektor	= fehler3(liste);
				fehler    		= fehlerVektor[0];
				anzFehler 		= (int)fehlerVektor[1];
				System.out.println("Nochmal: ---------------------" + anzFehler + " " + fehler);
				
//				if(alpha>0.0001){
					alpha = alpha*0.95;
//				}
//				else{
//					alpha = 1.5;
//				}
//				System.out.println("----------------------------- " + erfolg + " " + anzVb + " " + intervall);
				anzVb = 0;
			}
			
//			if(epoche%intervall==0){
//		
//				double erfolg = (double)anzVb/(double)intervall;
//				if(erfolg<1.0/5.0){
//					alpha = alpha*0.95;
//				}
//				else{
//					alpha = alpha/0.95;
//				}
//				System.out.println("----------------------------- " + erfolg + " " + anzVb + " " + intervall);
//				anzVb = 0;
//			}
*/
        }

//		for(int i=0;i<n;i++){
//			for(int j=0;j<n;j++){
//				w[i][j] = optAnzGewichte[i][j];
//			}
//		}
    }

//	public void trainierenStochastisch(double[][] liste) {
//
//		double[][] optGewichte = new double[n][n];
//		double klasse;
//		double fehler;
//		gewichteInitialisieren();
//
//		boolean stop  	 = false;
//		int anzFehler 	 = 0;
//		int minAnzFehler = Integer.MAX_VALUE;
//		int epoche    	 = 0;
//		boolean goBack 	 = false;
//		int zaehler 	 = 0;
//		
//		while (!stop) {
//			epoche++;
//			
//			int muster = (int)(Math.random()*liste.length);
//			for (int s = 0; s < liste.length; s++) {
//				if(muster > liste.length-1)muster=0;
//				eingabeSchichtInitialisieren(liste[muster]);
//				klasse = liste[muster][liste[muster].length - 1];
//				forward();
//				backward(klasse);
//				muster++;
//			}
//
//			double fehlerVektor[] = fehler3(liste);
//			fehler    			  = fehlerVektor[0];
//			anzFehler 			  = (int)fehlerVektor[1];
//			
//			if(anzFehler < minAnzFehler) {//neue Bestlösung
//				minAnzFehler = anzFehler;
//				for(int i=0;i<n;i++){
//					for(int j=0;j<n;j++){
//						optGewichte[i][j] = w[i][j];
//					}
//				}
//				alpha *= 1.1;
//		    }
//			else if(anzFehler > minAnzFehler){//Verschlechterung
//				for(int i=0;i<n;i++){
//					for(int j=0;j<n;j++){
//						w[i][j] = optGewichte[i][j];
//					}
//				}
//				alpha *= 0.9;
//			}
//			
//			System.out.println("-Epoche: " + epoche + " " + anzFehler + " " + fehler + " minAnzFehler " + minAnzFehler + " " + goBack + " " + alpha + " " + zaehler);
//			if (epoche >= maxEpoche || anzFehler == 0)	stop = true;
//		}
//	}

    /**
     * Forward-Pass
     */
    private void forward() {
        // Skip bias layer, so int layer = 1
        for (int layer = 1; layer < netz.length; layer++) {
            for (int nodeNumber = 0; nodeNumber < netz[layer].length; nodeNumber++) {
                int nodeNumberInCurrentLayer = netz[layer][nodeNumber];
                if (!bias[nodeNumberInCurrentLayer]) {
                    in[nodeNumberInCurrentLayer] = 0.0;
                    for (int nri = 0; nri < netz[layer - 1].length; nri++) {
                        int nodeNumberInPreviousLayer = netz[layer - 1][nri];
                        in[nodeNumberInCurrentLayer] += w[nodeNumberInPreviousLayer][nodeNumberInCurrentLayer] * a[nodeNumberInPreviousLayer];
                    }
                    a[nodeNumberInCurrentLayer] = aktivierungsFunktion(in[nodeNumberInCurrentLayer]);
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
        int outputLayer = netz.length - 1;
        for (int nrj = 0; nrj < netz[outputLayer].length; nrj++) {
            int outputNodeNumber = netz[outputLayer][nrj];
            final var error = a[outputNodeNumber] - klasse;
            delta[outputNodeNumber] = ableitungAktivierungsFunktion(in[outputNodeNumber]) * error;
        }
    }

    private void deltaHiddenLayers() {
        int ausgabeSchicht = netz.length - 1;

        for (int l = ausgabeSchicht - 1; l >= 0; l--) {
            for (int nri = 0; nri < netz[l].length; nri++) {
                int i = netz[l][nri];
                delta[i] = 0.0;
                if (!bias[i]) {
                    double sum = 0;
                    for (int nrj = 0; nrj < netz[l + 1].length; nrj++) {
                        int j = netz[l + 1][nrj];
                        sum += w[i][j] * delta[j];
                    }
                    delta[i] = ableitungAktivierungsFunktion(in[i]) * sum;
                }
            }
        }
    }

    private void updateWeights() {
        for (int l = 0; l < netz.length - 1; l++) {
            for (int nri = 0; nri < netz[l].length; nri++) {
                int i = netz[l][nri];

                for (int nrj = 0; nrj < netz[l + 1].length; nrj++) {
                    int j = netz[l + 1][nrj];
                    if (!bias[j]) {
                        final var gradient = a[i] * delta[j];
                        double delt = alpha * gradient;
                        w[i][j] -= delt;//Gradientenabstieg
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

//	private double aktivierungsFunktion(double x) {
//		if(x>=0)return x;
//		else    return 0;
//	}
//
//	private double ableitungAktivierungsFunktion(double x) {
//		if(x>=0)return 1;
//		else    return 0;
//	}
//	

    /**
     * Initialisierung
     */
    private void gewichteInitialisieren() {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                w[i][j] = 0;
            }
        }

        for (int l = 0; l < m - 1; l++) {
            for (int i = 0; i < netz[l].length; i++) {
                int indexi = netz[l][i];
                for (int j = 0; j < netz[l + 1].length; j++) {
                    int indexj = netz[l + 1][j];
                    if (!bias[indexj]) {
                        w[indexi][indexj] = Math.random();
                        if (Math.random() < 0.5)
                            w[indexi][indexj] = -w[indexi][indexj];
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
        for (int i = 0; i < netz.length - 1; i++) {// über alle Schichten
            int knoten = netz[i][0]; // der erste Knoten einer Schicht ist Bias!
            if (!bias[knoten])
                System.out.println("ups, nicht-Bias-Knoten als bias initialisiert");
            in[knoten] = 1.0;
            a[knoten] = 1.0;
        }

        // Alle Knoten der Eingabeschicht ab dem 2. Knoten mit Eingabe belegen (1. Knoten ist ja Bias!)
        // der letzte Wert in input ist der output und gehört nicht zur Eingabe, deshalb input.length -1
        for (int i = 0; i < input.length - 1; i++) {
            // in[0] ist Bias, deshalb i.ten Input bei bei in[i+1] speichern
            in[i + 1] = input[i];
            a[i + 1] = input[i];
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
            if ((a[n - 1] < 0.5 && (int) klasse == 1) || (a[n - 1] >= 0.5 && (int) klasse == 0)) anzFehler++;
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
            fehler += Math.pow(klasse - a[n - 1], 2);
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
            fehler[0] += Math.pow(klasse - a[n - 1], 2);
            if ((a[n - 1] < 0.5 && (int) klasse == 1) || (a[n - 1] >= 0.5 && (int) klasse == 0)) {
                fehler[1]++;
            }
        }
        return fehler;
    }

    /**
     * Methoden zur Evaluierung
     */
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
            if (a[n - 1] < 0.5 && (int) output == 1) {
                falschNegativ++;
                anzahlPositiv++;
            } else if (a[n - 1] >= 0.5 && (int) output == 1) {
                richtigPositiv++;
                anzahlPositiv++;
            } else if (a[n - 1] >= 0.5 && (int) output == 0) {
                falschPositiv++;
                anzahlNegativ++;
            } else if (a[n - 1] < 0.5 && (int) output == 0) {
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

        System.out.println("Anzahl Muster:  \t" + ergebnis[0]);
        System.out.println("Anzahl Positiv: \t" + ergebnis[1]);
        System.out.println("Anzahl Negativ: \t" + ergebnis[2]);
        System.out.println("Anteil Positiv: \t" + ergebnis[3]);
        System.out.println("Anteil Negativ: \t" + ergebnis[4]);

        System.out.println("Genauigkeit  :  \t" + ergebnis[5]);
        System.out.println("Trefferquote:   \t" + ergebnis[10]);
        System.out.println("Ausfallrate :   \t" + ergebnis[11]);

        System.out.println("richtigPositiv: \t" + ergebnis[6]);
        System.out.println("falsch Negativ: \t" + ergebnis[9]);
        System.out.println("richtigNegativ: \t" + ergebnis[8]);
        System.out.println("falsch Positiv: \t" + ergebnis[7]);
        return ergebnis;
    }

    /**
     * für GUI_Beispiele
     * zeigt grafisch den output des NN fuer alle Punkte (x1, x2) im angegebenen Intervall
     * zeigt die angegenben Daten zum Vergleich ebenfalls an
     * die Ausgabe wurde von einem Studenten entwickelt!
     */
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

    public int output(double[] x) {
        double[] input = new double[3];
        input[0] = x[0] / 100.;
        input[1] = x[1] / 100.;
        input[2] = 1.;//wird nicht benötigt

        eingabeSchichtInitialisieren(input);
        forward();

        double u = a[n - 1];
        int out;

        if (u < 0.5) out = 0;
        else out = +1;

        return out;
    }

    /*
     * Generierung des Vorlesungsbeispiel; hier können Forward/Backward geübt werden
     */

//		public void berechnungVorlesungsBeispiel(double[][] liste) {
//			//hiermit wurde der forward und backward-Pass beschrieben
//			double klasse;
//			gewichteInitialisieren();
//
//			w[0][4] = +0.1;
//			w[1][4] = -0.2;
//			w[2][4] = +0.3;
//			
//			w[0][5] = +0.4;
//			w[1][5] = -0.8;
//			w[2][5] = +0.7;
//			
//			w[3][6] = +0.3;
//			w[4][6] = -0.4;
//			w[5][6] = -0.2;
//			
//			
//			eingabeSchichtInitialisieren(liste[0]);
//			klasse = liste[0][liste[0].length - 1];
//
//			//this.ausgabeEingabeSchicht();
//			//this.ausgabeW();
//			
//			alpha = 1.0;
//			
//			for(int i=0;i<100;i++){
//				forward();
//				backward(klasse);
//				System.out.println(i + " " + a[n-1]);
//			}
//			
////			this.ausgabeInputwerte();
////			this.ausgabeKnotenwerte();
////			this.ausgabeDelta();
////			this.ausgabeW();
//		}

    /**
     * Methoden zur  Ausgabe der Netzparameter
     */
    public void ausgabeBias() {
        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < netz.length; l++) {
                if (i < netz[l].length) {
                    int k = netz[l][i];
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

    public void ausgabeNetzStruktur() {
        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < netz.length; l++) {
                if (i < netz[l].length) {
                    System.out.print(netz[l][i] + "\t\t");
                    ende = false;
                } else {
                    System.out.print("\t\t");
                }
            }
            System.out.println();
            i++;
        }
    }

    public void ausgabeKnotenwerte() {
        System.out.println("Ausgabe der Knoten-Ausgabewerte");

        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < netz.length; l++) {
                if (i < netz[l].length) {
                    int k = netz[l][i];
                    double preci = 1000;
                    double round = ((int) (a[k] * preci)) / preci;
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

    public void ausgabeDelta() {
        System.out.println("Ausgabe der DELTA");
        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < netz.length; l++) {
                if (i < netz[l].length) {
                    int k = netz[l][i];
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

    public void ausgabeInputwerte() {
        System.out.println("Ausgabe der Inputwerte");
        int i = 0;
        boolean ende = false;
        while (!ende) {
            ende = true;
            for (int l = 0; l < netz.length; l++) {
                if (i < netz[l].length) {
                    int k = netz[l][i];
                    double preci = 1000;
                    double round = ((int) (in[k] * preci)) / preci;
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

    public void ausgabeEingabeSchicht() {
        System.out.println("Ausgabe der Eingabeschicht");

        for (int i = 0; i < netz[0].length; i++) {
            int knoten = netz[0][i];
            System.out.print(a[knoten] + " ");
        }
        System.out.println();
    }

    public void ausgabeW() {
        System.out.println("Ausgabe der Gewichte");

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double preci = 1000;
                double round = ((int) (w[i][j] * preci)) / preci;
                if (round != 0)
                    System.out.println(i + " " + j + " " + round);
            }
            // System.out.println();
        }
        System.out.println();
    }


//		public void trainierenBatch(double[][] liste) {
//
//			double klasse;
//			double[] teilFehler;
//			double fehler       = 0.0;
//			int anzFehler 		= 0;
//			int minAnzFehler 	= Integer.MAX_VALUE;
//			boolean stop  		= false;
//			int epoche    		= 0;//hier nicht wirklich die Epoche!
//			int zaehler 		= 0;
//
//			gewichteInitialisieren();
//			
//			while (!stop) {
//				epoche++;
//				deltBatchInitialisieren();
//				
//				for (int s = 0; s < batchSize; s++) {
//					int muster = (int)(Math.random()*liste.length);
//					eingabeSchichtInitialisieren(liste[muster]);
//					klasse = liste[muster][liste[muster].length - 1];
//					forward();
//					deltBatchAktualisieren(klasse);
//				}
//				gewichteAktualisierenBatch(liste);
//			
//				teilFehler = fehler3(liste);
//				fehler     = teilFehler[0];
//				anzFehler  = (int)teilFehler[1]; 
//				if(anzFehler <= minAnzFehler) {//neue Bestlösung
//					minAnzFehler = anzFehler;
//			    }
//				System.out.println("-Epoche: " + epoche + " " + anzFehler + " - Fehler " + fehler + " minAnzFehler " + minAnzFehler + " " + alpha + " " + zaehler);
//				if (epoche >= maxEpoche || anzFehler == 0)	stop = true;
//			}
//		}
//		
//		public void deltBatchInitialisieren(){
//			for (int i = 0; i<n; i++) {
//				for (int j = 0; j<n; j++) {
//					gradient[i][j] = 0.0;	
//				}								
//			}
//		}
//		
//			
//		private void deltBatchAktualisieren(double klasse) {
//			
//			deltBatchAusgabeSchicht(klasse);
//
//			int ausgabeSchicht = netz.length - 1;
//			
//			for (int l = ausgabeSchicht - 1; l >= 0; l--) {
//				for (int nri = 0; nri < netz[l].length; nri++) {
//					int i = netz[l][nri];
//					delta[i] = 0.0;
//					if (!bias[i]) {
//						double sum = 0;
//						for (int nrj = 0; nrj < netz[l + 1].length; nrj++) {
//							int j = netz[l + 1][nrj];
//							sum  += w[i][j] * delta[j];
//						}
//						delta[i] = ableitungAktivierungsFunktion(in[i]) * sum;
//					}
//				}
//			}
//			//Gradientenanteil für Muster berechnen
//			for (int l = 0; l < netz.length - 1; l++) {
//				for (int nri = 0; nri < netz[l].length; nri++) {
//					int i = netz[l][nri];
//
//					for (int nrj = 0; nrj < netz[l + 1].length; nrj++) {
//						int j = netz[l + 1][nrj];
//						if (!bias[j]) {
//							double teilGradient = a[i] * delta[j];
//							gradient[i][j]      = gradient[i][j] + teilGradient;//Gradientenabstieg
//						}
//					}
//				}
//			}
//		}
//		private void deltBatchAusgabeSchicht(double klasse) {
//			int ausgabeSchicht = netz.length - 1;
//			for (int nrj = 0; nrj < netz[ausgabeSchicht].length; nrj++) {
//				int j 			= netz[ausgabeSchicht][nrj];
//				double yj 		= klasse;
//				delta[j]  		= ableitungAktivierungsFunktion(in[j]) * (a[j]-yj);
//			}
//		}

//		public void gewichteAktualisierenBatch(double[][] liste){
//			for (int l = 0; l < netz.length - 1; l++) {
//				for (int nri = 0; nri < netz[l].length; nri++) {
//					int i = netz[l][nri];
//					for (int nrj = 0; nrj < netz[l + 1].length; nrj++) {
//						int j = netz[l + 1][nrj];
//						if (!bias[j]) {
//							w[i][j] = w[i][j] - (1.0/batchSize)*alpha*gradient[i][j];//Gradientenabstieg
//						}
//					}
//				}
//			}
//		}
}
