import org.jblas.DoubleMatrix;

import java.util.Arrays;

public class KNNMatrix implements KNN {

    private final DoubleMatrix[] weights;
    private final DoubleMatrix[] output;
    private final DoubleMatrix[] delta;

    private final int[] layers;
    private final int maxEpoch;
    private final double bias;

    private final double maxAlpha;
    private final double minAlpha;
    private final double currentAlpha;

    public KNNMatrix(int dimensions, TestParameters testParameters) {
        maxAlpha = testParameters.getMaxAlpha();
        minAlpha = testParameters.getMinAlpha();
        currentAlpha = maxAlpha;
        maxEpoch = testParameters.getMaxEpoche();

        final var hiddenLayers = testParameters.getLayers();
        layers = new int[hiddenLayers.length + 1];
        System.arraycopy(hiddenLayers, 0, layers, 0, hiddenLayers.length);
        layers[layers.length - 1] = 2;

        weights = new DoubleMatrix[layers.length];
        var inputCount = dimensions;
        for (int layer = 0; layer < layers.length; layer++) {
            weights[layer] = DoubleMatrix.rand(layers[layer], inputCount);
            inputCount = layers[layer];
        }

        bias = testParameters.getBias();
        output = new DoubleMatrix[layers.length];
        delta = new DoubleMatrix[layers.length];
    }

    @Override
    public void trainieren(double[][] dataSet, boolean print) {
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            for (double[] data : dataSet) {
                forward(data);
                backward(data[data.length - 1]);
            }

            if (print) {
                double[] errorVector;
                errorVector = fehler3(dataSet);
                System.out.println("-Epoch: " + epoch + " " + (int) errorVector[1] + " " + String.format("%.4f", errorVector[0]));
            }

//            currentAlpha -= (maxAlpha - minAlpha) / maxEpoch;
        }
    }

    private void forward(double[] data) {
        DoubleMatrix activations;
        // forwards first layer only
        {
            int firstLayer = 0;
            //data contains result, which needs to be excluded
            final var inputData = new DoubleMatrix(Arrays.copyOfRange(data, 0, data.length - 1));
            activations = weights[firstLayer].mmul(inputData).add(bias);
            var newOutput = new double[layers[firstLayer]];
            for (int node = 0; node < activations.length; node++) {
                newOutput[node] = Functions.sigmuid(activations.get(node));
            }
            output[firstLayer] = new DoubleMatrix(newOutput);
        }

        //forward remaining layers
        {
            for (int layer = 1; layer < layers.length; layer++) {
                activations = weights[layer].mmul(output[layer - 1]).add(bias);
                var newOutput = new double[layers[layer]];

                for (int node = 0; node < activations.length; node++) {
                    newOutput[node] = Functions.sigmuid(activations.get(node)) + bias;
                }
                output[layer] = new DoubleMatrix(newOutput);
            }
        }
    }

    private double[] fehler3(double[][] liste) {
        double[] fehler = {0.0, 0.0};

        for (double[] doubles : liste) {
            forward(doubles);
            var classification = doubles[doubles.length - 1];
            final var outputLayer = output[output.length - 1];
            final var output = outputLayer.get(outputLayer.length - 1);
            fehler[0] += Math.pow(classification - output, 2);
            if ((output < 0.5 && (int) classification == 1) || (output >= 0.5 && (int) classification == 0)) {
                fehler[1]++;
            }
        }
        return fehler;
    }

    private void backward(double classification) {
        //backward delta last layer
        {
            var negativeClassification = -classification; // 1 or 0
            var lastLayer = layers.length - 1;

            var derivatives = new double[layers[lastLayer]];
            Arrays.setAll(derivatives, index -> Functions.sigmuidDerivative(output[lastLayer].get(index)));

            final var doubleMatrix = new DoubleMatrix(derivatives);
            final var mul = doubleMatrix.mul(output[lastLayer].add(negativeClassification));
            delta[lastLayer] = mul;
        }

        //backward delta remaining layers
        {
            for (int layer = layers.length - 2; layer >= 0; layer--) {
                var tmpDelta = new double[layers[layer]];
                int finalLayer = layer;
                Arrays.setAll(tmpDelta, index -> Functions.sigmuidDerivative(output[finalLayer].get(index)));

                final var mmul = delta[layer + 1].transpose().mmul(weights[layer + 1]);
                final var transpose = new DoubleMatrix(tmpDelta).transpose();
                delta[layer] = transpose.mul(mmul); //TODO
            }
        }

        //backwards update weights
        {
            for (int layer = 0; layer < layers.length; layer++) {
                final var mul1 = output[layer].mul(delta[layer].transpose());
                final var mul = mul1.mul(-currentAlpha);
                weights[layer].addiColumnVector(mul);
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
            var classification = lastLayer.get(lastLayer.length - 1);

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
