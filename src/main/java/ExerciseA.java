import java.io.File;

public class ExerciseA {

    private static final TestParameters[] testParameters = {
            new TestParameters(new int[]{20, 20}, 1, 0.01, 1000, 100, 10),
    };

    public static void main(String[] args) {
        System.out.println("start");
        var trainData = Einlesen.einlesenBossShit(new File("mnist_train_50k.csv"), true, 785);
        var testData = Einlesen.einlesenBossShit(new File("mnist_test_full.csv"), true, 785);

        int dimension = trainData[0].length - 1;

        for (TestParameters parameters : testParameters) {
            var start = System.currentTimeMillis();
            KNN netz = new KNNMatrix(dimension, parameters);

            netz.trainieren(trainData, false);

            var result = netz.evaluieren(testData);

            Utils.printParameters(parameters);
            Utils.printResult(result);
            System.out.println("time: " + (System.currentTimeMillis() - start));
        }
    }
}