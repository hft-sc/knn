import java.util.Arrays;

public class TestParameters {

    private final int[] layers;
    private final double alpha;
    private final int maxEpoche;

    public TestParameters(int[] layers, double alpha, int maxEpoche) {
        this.layers = layers;
        this.alpha = alpha;
        this.maxEpoche = maxEpoche;
    }

    public int[] getLayers() {
        return layers;
    }

    public double getAlpha() {
        return alpha;
    }

    public int getMaxEpoche() {
        return maxEpoche;
    }

    @Override
    public String toString() {
        return "{" +
                "layers=" + Arrays.toString(layers) +
                ", alpha=" + alpha +
                ", maxEpoche=" + maxEpoche +
                '}';
    }
}
