public class ExerciseA {

    private static final TestParameters[] testParameters = {
            new TestParameters(new int[]{}, 1, 1, 2000, 0),
    };
    private static final double[][] XOR_TRAIN = new double[][]{
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {0, 0, 0},
            {1, 1, 0},
            {1, 1, 0},
            {1, 1, 0},
            {1, 1, 0},
            {1, 1, 0},
            {0, 1, 1},
            {0, 1, 1},
            {0, 1, 1},
            {0, 1, 1},
            {1, 0, 1},
            {1, 0, 1},
            {1, 0, 1},
            {1, 0, 1},
            {1, 0, 1},
            {1, 0, 1},
            {1, 0, 1}
    };
    public static final double[][] XOR_TEST = {
            {0, 1, 1},
            {0, 1, 1},
            {1, 0, 1},
            {0, 1, 1},
            {0, 1, 1},
            {0, 0, 0},
            {0, 0, 0},
            {1, 1, 0},
            {1, 1, 0},
            {1, 1, 0}
    };

    public static void main(String[] args) {
        System.out.println("start");
//        var trainData = Einlesen.einlesenBankdaten(new File("train_10k.csv"), false);
//        var testData = Einlesen.einlesenBankdaten(new File("test_10k.csv"), false);
        var trainData = XOR_TRAIN;
        var testData = XOR_TEST;
//        var trainData = Einlesen.einlesenDiabetes(new File("diabetes_train.csv"), true, false);
//        var testData = Einlesen.einlesenDiabetes(new File("diabetes_test.csv"), false, false);
        int dimension = trainData[0].length - 1;

        for (TestParameters parameters : testParameters) {
            {
                var start = System.currentTimeMillis();
                KNN netz = new KNNMatrix(dimension, parameters);

                netz.trainieren(trainData, false);//Verlustfunktion min

                var result = netz.evaluieren(testData);
                Utils.printResult(parameters, result);
                System.out.println("time: " + (System.currentTimeMillis() - start));
            }

//            {
//                var start = System.currentTimeMillis();
//                KNN netz = new KNNImpl(dimension, parameters);
//
//                netz.trainieren(trainData, false);//Verlustfunktion min
//
//                var result = netz.evaluieren(testData);
//                Utils.printResult(parameters, result);
//                System.out.println("time: " + (System.currentTimeMillis() - start));
//            }

        }


    }
}