import java.util.Arrays;

public class TestParameters {

    private final int[] layers;
    private final double maxAlpha;
    private final double minAlpha;
    private final int maxEpoch;
    private final int batchSize;
    private final int resultCount;


    public TestParameters(int[] layers, double maxAlpha, double minAlpha, int maxEpoch, int batchSize, int resultCount) {
        this.layers = layers;
        this.maxAlpha = maxAlpha;
        this.minAlpha = minAlpha;
        this.maxEpoch = maxEpoch;
        this.batchSize = batchSize;
        this.resultCount = resultCount;
    }

    public int[] getLayers() {
        return layers;
    }

    public double getMaxAlpha() {
        return maxAlpha;
    }

    public double getMinAlpha() {
        return minAlpha;
    }

    public int getMaxEpoch() {
        return maxEpoch;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getResultCount() {
        return resultCount;
    }

    @Override
    public String toString() {
        return "{" +
                "layers=" + Arrays.toString(layers) +
                ", maxAlpha=" + maxAlpha +
                ", minAlpha=" + minAlpha +
                ", maxEpoche=" + maxEpoch +
                ", batchSize=" + batchSize +
                '}';
    }
}
