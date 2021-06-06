import java.io.File;

public class ExerciseA {

    private static final TestParameters[] testParameters = {
            new TestParameters(new int[]{8}, 1, 1, 1000),
            new TestParameters(new int[]{8}, 1, 1, 10000),
            new TestParameters(new int[]{8}, 10, 1, 1000),
            new TestParameters(new int[]{8}, 50, 1, 1000),
    };

    public static void main(String[] args) {
//        double[][] daten = Einlesen.einlesenBankdaten(new File("4_Trainingsdaten.csv"), false);
        double[][] daten = Einlesen.einlesenDiabetes(new File("diabetes_train.csv"), true, false);
        int dimension = daten[0].length - 1;

        for (TestParameters parameters : testParameters) {
            KNN netz = new KNN(dimension, parameters.getLayers(), parameters.getMaxAlpha(), parameters.getMinAlpha(), parameters.getMaxEpoche());

            netz.trainieren(daten, false);//Verlustfunktion min

//            daten = Einlesen.einlesenBankdaten(new File("test01.csv"), false);
            daten = Einlesen.einlesenDiabetes(new File("diabetes_test.csv"), false, false);
            var result = netz.evaluieren(daten);
            Utils.printResult(parameters, result);
        }


    }
}