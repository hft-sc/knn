import java.util.Arrays;

public class TestParameters {

    private final int[] layers;
    private final double maxAlpha;
    private final double minAlpha;
    private final int maxEpoche;
    private final double bias;

    public TestParameters(int[] layers, double maxAlpha, double minAlpha, int maxEpoche, double bias) {
        this.layers = layers;
        this.maxAlpha = maxAlpha;
        this.minAlpha = minAlpha;
        this.maxEpoche = maxEpoche;
        this.bias = bias;
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

    public int getMaxEpoche() {
        return maxEpoche;
    }

    public double getBias() {
        return bias;
    }

    @Override
    public String toString() {
        return "{" +
                "layers=" + Arrays.toString(layers) +
                ", maxAlpha=" + maxAlpha +
                ", minAlpha=" + minAlpha +
                ", maxEpoche=" + maxEpoche +
                ", bias=" + bias +
                '}';
    }
}
