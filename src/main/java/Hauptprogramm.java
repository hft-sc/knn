import java.io.File;

public class Hauptprogramm {

    public static void main(String[] args) {
//		double[][] daten = Einlesen.einlesenBankdaten(new File("4_Trainingsdaten.csv"));
//		double[][] daten = Einlesen.einlesenDiabetes(new File("diabetes.csv"), true);
//		double[][] daten = Einlesen.einlesenVorlesungsbeispiele(new File("svmKnearestNick.txt"));
        double[][] daten = Einlesen.einlesenVorlesungsbeispiele(new File("wetter.txt"));
//		double[][] daten = Einlesen.einlesenVorlesungsbeispiele(new File("XOR.txt"));
        int dimension = daten[0].length - 1;

        //Einlesen.auslesen(daten);

        int[] strukturNN = {5};//anzahl Knoten (incl. Bias) pro Hiddenschicht
        KNN netz = new KNN(dimension, strukturNN, 0.5, 0.5, 10);

        netz.trainieren(daten, true);//Verlustfunktion min
        //netz.trainierenStochastisch(daten);
        //netz.trainierenMiniBatch(daten);
        //netz.trainierenBatch(daten);

//		daten = Einlesen.einlesenBankdaten(new File("test01.csv")); 	
//		daten = Einlesen.einlesenBankdaten(new File("5_Testdaten.csv"));
//		daten = Einlesen.einlesenDiabetes(new File("diabetes.csv"), false);
        //Einlesen.auslesen(daten);
//		var result = netz.evaluieren(daten);
//        Utils.printResult(result);


        netz.evaluierenGUIII(daten);
    }
}
