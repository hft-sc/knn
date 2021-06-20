import org.apache.commons.collections4.ListUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class KNNMatrix implements KNN {

    /**
     * array of n x m matrix
     * each entry of array is one layer
     */
    private final DoubleMatrix[] weights;

    /**
     * array of vector
     * each entry of array is one layer
     */
    private final DoubleMatrix[] biases;

    private final int[] layers;
    private final int maxEpoch;
    private final int batchSize;

    private final double maxAlpha;
    private final double minAlpha;
    private double currentAlpha;

    public KNNMatrix(int dimensions, TestParameters testParameters) {
        maxAlpha = testParameters.getMaxAlpha();
        minAlpha = testParameters.getMinAlpha();
        currentAlpha = maxAlpha;
        maxEpoch = testParameters.getMaxEpoch();
        batchSize = testParameters.getBatchSize();

        final var hiddenLayers = testParameters.getLayers();
        layers = new int[hiddenLayers.length + 2];
        layers[0] = dimensions;
        System.arraycopy(hiddenLayers, 0, layers, 1, hiddenLayers.length);
        layers[layers.length - 1] = 1; //auf 10 Knoten

        weights = new DoubleMatrix[layers.length];
        for (int layer = 0; layer < layers.length - 1; layer++) {
            final var nodeCount = layers[layer];
            final var nodeCountNextLayer = layers[layer + 1];
            weights[layer] = DoubleMatrix.rand(nodeCountNextLayer, nodeCount);
        }

        biases = new DoubleMatrix[layers.length];
        for (int layer = 0; layer < layers.length; layer++) {
            biases[layer] = DoubleMatrix.rand(layers[layer], 1);
        }
        System.out.println();
    }

    @Override
    public void trainieren(double[][] dataSet, boolean print) {
        for (int epoch = 0; epoch < maxEpoch; epoch++) {
            final var shuffledDataSet = Arrays.asList(dataSet);
            Collections.shuffle(shuffledDataSet);
            final var batches = ListUtils.partition(shuffledDataSet, batchSize);
            for (List<double[]> batch : batches) {
                var biasAdjustments = new LinkedList<DoubleMatrix[]>();
                var weightAdjustments = new LinkedList<DoubleMatrix[]>();

                for (double[] row : batch) {
                    var values = forward(row);
                    var adjustments = calculateAdjustments(row, values);
                    biasAdjustments.add(adjustments.getLeft());
                    weightAdjustments.add(adjustments.getRight());
                }

                for (DoubleMatrix[] adjustment : biasAdjustments) {
                    adjustBiases(adjustment);
                }
                for (DoubleMatrix[] adjustment : weightAdjustments) {
                    adjustWeights(adjustment);
                }
            }

            if (print && epoch % 100 == 0) {
                double[] errorVector;
                errorVector = fehler3(dataSet);
                System.out.println("-Epoch: " + epoch + "fNeg " + (int) errorVector[1] + " fPos " + (int) errorVector[2]
                        + " " + String.format("%.4f", errorVector[0]) + " alpha " + currentAlpha);
                if ((int) errorVector[1] == 0 && (int) errorVector[2] == 0) break;
            }

            currentAlpha = Math.max(minAlpha, currentAlpha - (maxAlpha - minAlpha) / maxEpoch);
        }
    }

    /**
     * Left are the zs, right are the activations
     */
    private Pair<DoubleMatrix[], DoubleMatrix[]> forward(double[] row) {
        var zs = new DoubleMatrix[layers.length];
        var activations = new DoubleMatrix[layers.length];
        for (int layer = 0; layer < layers.length; layer++) {
            activations[layer] = new DoubleMatrix(layers[layer]);
        }

        activations[0] = new DoubleMatrix(Arrays.copyOfRange(row, 0, row.length - 1));

        for (int layer = 1; layer < layers.length; layer++) {
            zs[layer] = weights[layer - 1]
                    .mmul(activations[layer - 1])
                    .addiColumnVector(biases[layer]);
            for (int i = 0; i < zs[layer].length; i++) {
                final var sigmoid = Functions.sigmoid(zs[layer].get(i));
                activations[layer].put(i, sigmoid);
            }
        }

        return Pair.of(zs, activations);
    }

    private double[] fehler3(double[][] liste) { //Anpassen
        double[] fehler = {0.0, 0.0, 0.0};

        for (double[] doubles : liste) {
            var values = forward(doubles);
            var zs = values.getLeft();

            var expected = doubles[doubles.length - 1];
            final var outputLayer = zs[zs.length - 1];
            final var output = outputLayer.get(outputLayer.length - 1);
            fehler[0] += Math.pow(expected - output, 2);
            if (output < 0.5 && (int) expected == 1) {
                fehler[1]++;
            } else if (output >= 0.5 && (int) expected == 0) {
                fehler[2]++;
            }
        }
        return fehler;
    }

    private Pair<DoubleMatrix[], DoubleMatrix[]> calculateAdjustments(double[] row, Pair<DoubleMatrix[], DoubleMatrix[]> values) {
        var zs = values.getLeft();
        var activations = values.getRight();

        var outputLayer = layers.length - 1;
        var expected = row[row.length - 1];

        var delta = activations[activations.length - 1] //erweitern um 10
                .add(-expected)
                .muli(Functions.sigmoidDerivative(zs[outputLayer]));

        DoubleMatrix[] biasAdjustments = new DoubleMatrix[biases.length];
        biasAdjustments[biasAdjustments.length - 1] = delta;
        DoubleMatrix[] weightAdjustments = new DoubleMatrix[weights.length];
        weightAdjustments[weightAdjustments.length - 1] = delta.mmul(activations[weightAdjustments.length - 2].transpose());

        for (int layer = layers.length - 2; layer > 0; layer--) {
            var z = zs[layer];
            var sp = Functions.sigmoidDerivative(z);
            delta = weights[layer].transpose()
                    .mmul(delta)
                    .muli(sp);
            biasAdjustments[layer] = delta;
            weightAdjustments[layer] = delta.mmul(activations[layer - 1].transpose());
        }

        return Pair.of(biasAdjustments, weightAdjustments);
    }

    private void adjustWeights(DoubleMatrix[] weightAdjustments) {
        for (int layer = 1; layer < weights.length; layer++) {
            weights[layer - 1].addi(weightAdjustments[layer].muli(-1));
        }
    }

    private void adjustBiases(DoubleMatrix[] biasAdjustments) {
        for (int layer = 1; layer < biases.length; layer++) {
            biases[layer].addi(biasAdjustments[layer].muli(-1));
        }
    }

    @Override
    public double[] evaluieren(double[][] liste) {
        double[] result = new double[12];

        int falschPositiv = 0;
        int falschNegativ = 0;
        int richtigPositiv = 0;
        int richtigNegativ = 0;
        int anzahlPositiv = 0;
        int anzahlNegativ = 0;

        for (double[] data : liste) {
            var values = forward(data);
            var zs = values.getLeft();
            var lastLayer = zs[zs.length - 1];
            var classification = lastLayer.get(0);

            double expectedOutput = data[data.length - 1];
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
