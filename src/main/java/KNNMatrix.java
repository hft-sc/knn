import org.jblas.DoubleMatrix;

import java.util.Arrays;

public class KNNMatrix implements KNN {

    private final DoubleMatrix[] weight;
    private final double[] bias;
    private final DoubleMatrix[] input;
    private final DoubleMatrix[] output;
    private final DoubleMatrix[] delta;

    private final int[] layers;
    private final int maxEpoch;
    private final int dimensions;

    private final double maxAlpha;
    private final double minAlpha;
    private double currentAlpha;

    public KNNMatrix(int dimensions, TestParameters testParameters) {
        maxAlpha = testParameters.getMaxAlpha();
        minAlpha = testParameters.getMinAlpha();
        currentAlpha = maxAlpha;
        maxEpoch = testParameters.getMaxEpoche();
        this.dimensions = dimensions;

        final var hiddenLayers = testParameters.getLayers();
        layers = new int[hiddenLayers.length + 2];
        layers[0] = dimensions;
        System.arraycopy(hiddenLayers, 0, layers, 1, hiddenLayers.length);
        layers[layers.length - 1] = 1;

        weight = new DoubleMatrix[layers.length];
        for (int layer = 0; layer < layers.length - 1; layer++) {
            DoubleMatrix.rand(layers[layer], layers[layer + 1]);
        }
        Arrays.setAll(weight, index -> DoubleMatrix.rand(layers[index], layers[index + 1]));

        bias = new double[layers.length - 1];
        Arrays.fill(bias, testParameters.getBias());

        input = new DoubleMatrix[layers.length - 1];
//        for (int layer = 0; layer < layers.length; layer++) {
//            input[layer] = new DoubleMatrix(layers[layer]);
//        }

        output = new DoubleMatrix[layers.length - 1];
//        for (int layer = 0; layer < layers.length; layer++) {
//            output[layer] = new DoubleMatrix(layers[layer]);
//        }

        delta = new DoubleMatrix[layers.length - 1];
    }

    @Override
    public void trainieren(double[][] dataSet, boolean print) {
        for (int epoch = 0; epoch < maxEpoch; epoch++) {

            for (double[] data : dataSet) {
                forward(data);
                backward(data[data.length - 1]);
            }

            currentAlpha -= (maxAlpha - minAlpha) / maxEpoch;
        }

    }

    private void backward(double datum) {
        //backward delta last layer
        {
            var negativeClassification = -datum; // 1 or 0
            var lastLayer = layers.length - 1;
            var tmpDelta = new double[layers[lastLayer]];
            Arrays.setAll(tmpDelta, index -> Functions.sigmuidDerivative(input[lastLayer].get(index)));
            delta[lastLayer] = new DoubleMatrix(tmpDelta).mul(output[lastLayer].add(negativeClassification));
        }

        //backward delta remaining layers
        {
            for (int layer = layers.length - 1; layer >= 0; layer--) {
                var tmpDelta = new double[layers[layer]];
                int finalLayer = layer;
                Arrays.setAll(tmpDelta, index -> Functions.sigmuidDerivative(input[finalLayer].get(index)));
                delta[layer] = new DoubleMatrix(tmpDelta).mul(weight[layer].mmul(delta[layer + 1])); //TODO
            }
        }


        //backwards update weights
        {
            for (int layer = 0; layer < layers.length - 1; layer++) {
                weight[layer].addi(output[layer].mul(delta[layer + 1]).mul(-currentAlpha));
            }
        }
    }

    private void forward(double[] data) {
        // forwards first layer only
        {
            int firstLayer = 0;
            //data contains result, which needs to be excluded
            final var inputData = new DoubleMatrix(Arrays.copyOfRange(data, 0, data.length - 1));
            input[firstLayer] = inputData.transpose().mmul(weight[firstLayer]);
            var newOutput = new double[layers[firstLayer]];
            Arrays.setAll(newOutput, index -> Functions.sigmuid(input[firstLayer].get(index)));
            output[firstLayer] = new DoubleMatrix(newOutput);
        }

        //forward remaining layers
        {
            for (int layer = 1; layer < layers.length; layer++) {
                input[layer] = weight[layer].mmul(output[layer - 1]);
                var newOutput = new double[layers[layer]];
                int finalLayer = layer;
                Arrays.setAll(newOutput, index -> Functions.sigmuid(input[finalLayer].get(index)));
                output[layer] = new DoubleMatrix(newOutput);
            }
        }
    }

    @Override
    public double[] evaluieren(double[][] liste) {
        double[] result = new double[12];

        double expectedOutput;
        int falschPositiv = 0;
        int falschNegativ = 0;
        int richtigPositiv = 0;
        int richtigNegativ = 0;
        int anzahlPositiv = 0;
        int anzahlNegativ = 0;

        for (double[] data : liste) {
            forward(data);
            var lastLayer = output[output.length - 1];
            var classification = lastLayer.get(lastLayer.length);

            expectedOutput = data[data.length - 1];
            if ((int) expectedOutput == 1) {
                anzahlPositiv++;

                if (classification < 0.5) {
                    falschNegativ++;
                } else {
                    richtigPositiv++;
                }

            } else if ((expectedOutput == 0)) {
                anzahlNegativ++;

                if (classification < 0.5) {
                    richtigNegativ++;
                } else {
                    falschPositiv++;
                }

            } else {
                System.out.println("nonono");
            }
        }

        if (anzahlPositiv != richtigPositiv + falschNegativ) System.out.println("Error1 in Auswertung");
        if (anzahlNegativ != richtigNegativ + falschPositiv) System.out.println("Error2 in Auswertung");
        if (anzahlPositiv + anzahlNegativ != liste.length) System.out.println("Error3 in Auswertung");

        result[0] = liste.length;
        result[1] = anzahlPositiv;
        result[2] = anzahlNegativ;
        result[3] = (double) anzahlPositiv / (double) liste.length;
        result[4] = (double) anzahlNegativ / (double) liste.length;
        result[5] = (double) (richtigPositiv + richtigNegativ) / (double) liste.length;
        result[6] = richtigPositiv;
        result[7] = falschPositiv;
        result[8] = richtigNegativ;
        result[9] = falschNegativ;
        result[10] = (double) richtigPositiv / (double) (richtigPositiv + falschNegativ);
        result[11] = (double) falschPositiv / (double) (richtigNegativ + falschPositiv);

        return result;
    }
}
