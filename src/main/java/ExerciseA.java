import java.io.File;

public class ExerciseA {

    private static final TestParameters[] testParameters = {
            new TestParameters(new int[]{8}, 10, 1, TestParameters.AlphaModifierType.LINEAR, 1000),
    };

    public static void main(String[] args) {
//        double[][] daten = Einlesen.einlesenBankdaten(new File("4_Trainingsdaten.csv"), false);
        double[][] daten = Einlesen.einlesenDiabetes(new File("diabetes_train.csv"), true, false);
        int dimension = daten[0].length - 1;

        for (TestParameters parameters : testParameters) {
            KNN netz = new KNNImpl(dimension, parameters);

            netz.trainieren(daten, false);//Verlustfunktion min

//            daten = Einlesen.einlesenBankdaten(new File("test01.csv"), false);
            daten = Einlesen.einlesenDiabetes(new File("diabetes_test.csv"), false, false);
            var result = netz.evaluieren(daten);
            Utils.printResult(parameters, result);
        }


    }
}