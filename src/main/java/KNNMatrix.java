import org.jblas.DoubleMatrix;

import java.util.ArrayList;
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
    private double currentAlpha;

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
        for (int layer = 0; layer < layers.length; layer++) {
            output[layer] = DoubleMatrix.rand(layers[layer], 1);
        }

        delta = new DoubleMatrix[layers.length];
        for (int layer = 0; layer < layers.length; layer++) {
            delta[layer] = DoubleMatrix.rand(layers[layer], 1);
        }

    }

    @Override
    public void trainieren(double[][] dataSet, boolean print) {
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            for (double[] data : dataSet) {
                forward(data);
                backward(data);
            }

            if (print) {
                double[] errorVector;
                errorVector = fehler3(dataSet);
                System.out.println("-Epoch: " + epoch + "fNeg " + (int) errorVector[1] + " fPos " + (int) errorVector[2]
                        + " " + String.format("%.4f", errorVector[0]));
            }

            currentAlpha -= (maxAlpha - minAlpha) / maxEpoch;
        }
    }

    private void forward(double[] row) {
        var inputs = new DoubleMatrix(Arrays.copyOfRange(row, 0, row.length - 1));

        for (int layer = 0; layer < layers.length; layer++) {
            final var nodeCount = layers[layer];
            DoubleMatrix newInputs = new DoubleMatrix(nodeCount, 1);
            for (int node = 0; node < nodeCount; node++) {
                var activation = active(weights[layer].getRow(node), inputs, bias);
                final var outputValue = Functions.sigmuid(activation);
                output[layer].put(node, outputValue);
                newInputs.put(node, outputValue);
            }
            inputs = newInputs;
        }
    }

    /**
     * For a single node
     */
    private double active(DoubleMatrix weights, DoubleMatrix inputs, double bias) {
        var activation = bias;
        for (int i = 0; i < inputs.length; i++) {
            activation += weights.get(i) * inputs.get(i);
        }
        return activation;
    }

    private void backward(double[] row) {
        var expected = row[row.length - 1];
        double[] expectedArray = expected == 1.0 ? new double[]{0, 1.0} : new double[]{1.0, 0};
        backwardErrors(expectedArray);

        updateWeights(row);
    }

    private void backwardErrors(double[] expected) {
        var errors = new ArrayList<Double>();

        for (int layer = layers.length - 1; layer >= 0; layer--) {
            if (layer == layers.length - 1) {
                for (int node = 0; node < layers[layer]; node++) {
                    errors.add(expected[node] - output[layer].get(node));
                }
            } else {
                for (int node = 0; node < layers[layer]; node++) {
                    var error = 0.0;
                    for (int nodeNextLayer = 0; nodeNextLayer < layers[layer + 1]; nodeNextLayer++) {
                        error += delta[layer + 1].get(nodeNextLayer) * weights[layer + 1].get(nodeNextLayer, node);
                    }
                    errors.add(error);
                }
            }

            for (int node = 0; node < layers[layer]; node++) {
                delta[layer].put(node, errors.get(node) * Functions.sigmuidDerivative(output[layer].get(node)));
            }
        }
    }

    private void updateWeights(double[] row) {
        for (int layer = 0; layer < layers.length; layer++) {
            var inputs = new DoubleMatrix(Arrays.copyOfRange(row, 0, row.length - 1));
            if (layer != 0) {
                inputs = output[layer - 1];
            }

            for (int node = 0; node < layers[layer]; node++) {
                for (int input = 0; input < inputs.length; input++) {
                    var newValue = weights[layer].get(node, input) + currentAlpha * delta[layer].get(node) * inputs.get(input);
                    weights[layer].put(node, input, newValue);
                }
            }
        }
    }

    private double[] fehler3(double[][] liste) {
        double[] fehler = {0.0, 0.0, 0.0};

        for (double[] doubles : liste) {
            forward(doubles);
            var classification = doubles[doubles.length - 1];
            final var outputLayer = output[output.length - 1];
            final var output = outputLayer.get(outputLayer.length - 1);
            fehler[0] += Math.pow(classification - output, 2);
            if (output < 0.5 && (int) classification == 1) {
                fehler[1]++;
            } else if (output >= 0.5 && (int) classification == 0) {
                fehler[2]++;
            }
        }
        return fehler;
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
