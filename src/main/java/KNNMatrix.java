import org.apache.commons.collections4.ListUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;

import java.util.*;

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
        layers[layers.length - 1] = testParameters.getResultCount(); //auf 10 Knoten

        var random = new Random();
        weights = new DoubleMatrix[layers.length];
        for (int layer = 0; layer < layers.length - 1; layer++) {
            final var nodeCount = layers[layer];
            final var nodeCountNextLayer = layers[layer + 1];
//            weights[layer] = DoubleMatrix.rand(nodeCountNextLayer, nodeCount);
            weights[layer] = new DoubleMatrix(nodeCountNextLayer, nodeCount);
            for (int node = 0; node < weights[layer].length; node++) {
                weights[layer].put(node, random.nextDouble() * 2 - 1);
            }
        }

        biases = new DoubleMatrix[layers.length];
        for (int layer = 0; layer < layers.length; layer++) {
            biases[layer] = new DoubleMatrix(layers[layer], 1);
            for (int node = 0; node < biases[layer].length; node++) {
                biases[layer].put(node, random.nextDouble() * 2 - 1);
            }
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

                var totalBiasAdjustments = biasAdjustments.get(0);
                for (int i = 1; i < biasAdjustments.size(); i++) {
                    for (int j = 1; j < totalBiasAdjustments.length; j++) {
                        totalBiasAdjustments[j].addi(biasAdjustments.get(i)[j]);
                    }
                }
                adjustBiases(totalBiasAdjustments);

                var totalWeightAdjustments = weightAdjustments.get(0);
                for (int i = 1; i < weightAdjustments.size(); i++) {
                    for (int j = 1; j < totalWeightAdjustments.length; j++) {
                        totalWeightAdjustments[j].addi(weightAdjustments.get(i)[j]);
                    }
                }
                adjustWeights(totalWeightAdjustments);
            }

            if (print && epoch % 5 == 0) {
                double[] errorVector;
                errorVector = evaluieren(dataSet);
                System.out.print("Epoch: " + epoch + "  ");
                Utils.printResult(errorVector);
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

    private Pair<DoubleMatrix[], DoubleMatrix[]> calculateAdjustments(double[] row, Pair<DoubleMatrix[], DoubleMatrix[]> values) {
        var zs = values.getLeft();
        var activations = values.getRight();

        var outputLayer = layers.length - 1;
        double[] expected = new double[10];
        expected[(int) row[row.length - 1]] = 1;


        // var expected = row[row.length - 1];
        DoubleMatrix expectedMatrix = new DoubleMatrix(expected);
        var delta = activations[activations.length - 1]
                .add(expectedMatrix.neg())
                .muli(Functions.sigmoidDerivative(zs[outputLayer]));

        DoubleMatrix[] biasAdjustments = new DoubleMatrix[biases.length];
        biasAdjustments[biasAdjustments.length - 1] = delta;
        DoubleMatrix[] weightAdjustments = new DoubleMatrix[weights.length];
        weightAdjustments[weightAdjustments.length - 1] = delta.mmul(activations[weightAdjustments.length - 2].
                transpose());

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
            weights[layer - 1].addi(weightAdjustments[layer].mul(-currentAlpha / batchSize));
        }
    }

    private void adjustBiases(DoubleMatrix[] biasAdjustments) {
        for (int layer = 1; layer < biases.length; layer++) {
            biases[layer].addi(biasAdjustments[layer].mul(-currentAlpha / batchSize));
        }
    }

    @Override
    public double[] evaluieren(double[][] liste) {
        double[] result = new double[25];

        int nullRichtig = 0;
        int einsRichtig = 0;
        int zweiRichtig = 0;
        int dreiRichtig = 0;
        int vierRichtig = 0;
        int fuenfRichtig = 0;
        int sechsRichtig = 0;
        int siebenRichtig = 0;
        int achtRichtig = 0;
        int neunRichtig = 0;

        int nullFalsch = 0;
        int einsFalsch = 0;
        int zweiFalsch = 0;
        int dreiFalsch = 0;
        int vierFalsch = 0;
        int fuenfFalsch = 0;
        int sechsFalsch = 0;
        int siebenFalsch = 0;
        int achtFalsch = 0;
        int neunFalsch = 0;

        for (double[] data : liste) {
            var values = forward(data);
            var activations = values.getRight();
            var lastLayer = activations[activations.length - 1];

            double expectedOutput = data[data.length - 1];
            final var actualOutput = lastLayer.argmax();
            if (expectedOutput == actualOutput) {
                if (0 == actualOutput) {
                    nullRichtig++;
                } else if (1 == actualOutput) {
                    einsRichtig++;
                } else if (2 == actualOutput) {
                    zweiRichtig++;
                } else if (3 == actualOutput) {
                    dreiRichtig++;
                } else if (4 == actualOutput) {
                    vierRichtig++;
                } else if (5 == actualOutput) {
                    fuenfRichtig++;
                } else if (6 == actualOutput) {
                    sechsRichtig++;
                } else if (7 == actualOutput) {
                    siebenRichtig++;
                } else if (8 == actualOutput) {
                    achtRichtig++;
                } else if (9 == actualOutput) {
                    neunRichtig++;
                }
            } else {
                if (0 == expectedOutput) {
                    nullFalsch++;
                } else if (1 == expectedOutput) {
                    einsFalsch++;
                } else if (2 == expectedOutput) {
                    zweiFalsch++;
                } else if (3 == expectedOutput) {
                    dreiFalsch++;
                } else if (4 == expectedOutput) {
                    vierFalsch++;
                } else if (5 == expectedOutput) {
                    fuenfFalsch++;
                } else if (6 == expectedOutput) {
                    sechsFalsch++;
                } else if (7 == expectedOutput) {
                    siebenFalsch++;
                } else if (8 == expectedOutput) {
                    achtFalsch++;
                } else if (9 == expectedOutput) {
                    neunFalsch++;
                }
            }
        }

        result[0] = liste.length;
        result[1] = nullRichtig;
        result[2] = einsRichtig;
        result[3] = zweiRichtig;
        result[4] = dreiRichtig;
        result[5] = vierRichtig;
        result[6] = fuenfRichtig;
        result[7] = sechsRichtig;
        result[8] = siebenRichtig;
        result[9] = achtRichtig;
        result[10] = neunRichtig;
        result[11] = nullFalsch;
        result[12] = einsFalsch;
        result[13] = zweiFalsch;
        result[14] = dreiFalsch;
        result[15] = vierFalsch;
        result[16] = fuenfFalsch;
        result[17] = sechsFalsch;
        result[18] = siebenFalsch;
        result[19] = achtFalsch;
        result[20] = neunFalsch;
        result[21] = (double) (nullRichtig + einsRichtig + zweiRichtig + dreiRichtig + vierRichtig + fuenfRichtig
                + sechsRichtig + siebenRichtig + achtRichtig + neunRichtig) / (double) liste.length;
        result[22] = (double) (nullFalsch + einsFalsch + zweiFalsch + dreiFalsch + vierFalsch + fuenfFalsch
                + sechsFalsch + siebenFalsch + achtFalsch + neunFalsch) / (double) liste.length;


        return result;
    }
}
