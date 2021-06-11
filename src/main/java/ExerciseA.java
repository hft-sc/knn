import java.io.File;

public class ExerciseA {

    private static final TestParameters[] testParameters = {
            new TestParameters(new int[]{8}, 10, 1, 1000),
            new TestParameters(new int[]{8}, 10, 1, 10000),
    };

    public static void main(String[] args) {
        double[][] daten = Einlesen.einlesenBankdaten(new File("train_10k.csv"), false);
//        double[][] daten = Einlesen.einlesenDiabetes(new File("diabetes_train.csv"), true, false);
        int dimension = daten[0].length - 1;

        for (TestParameters parameters : testParameters) {
            var start = System.currentTimeMillis();
            KNN netz = new KNNxrl(dimension, parameters);

            netz.trainieren(daten, false);//Verlustfunktion min

            daten = Einlesen.einlesenBankdaten(new File("test_10k.csv"), false);
//            daten = Einlesen.einlesenDiabetes(new File("diabetes_test.csv"), false, false);
            var result = netz.evaluieren(daten);
            Utils.printResult(parameters, result);
            System.out.println("time: " + (System.currentTimeMillis() - start));
        }


    }
}