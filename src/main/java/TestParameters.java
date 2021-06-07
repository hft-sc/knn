import java.util.Arrays;

public class TestParameters {

    private final int[] layers;
    private final double maxAlpha;
    private final double minAlpha;
    private final AlphaModifierType alphaModifierType;
    private final int maxEpoch;

    public TestParameters(int[] layers, double maxAlpha, double minAlpha, AlphaModifierType alphaModifierType, int maxEpoche) {
        this.layers = layers;
        this.maxAlpha = maxAlpha;
        this.minAlpha = minAlpha;
        this.alphaModifierType = alphaModifierType;
        this.maxEpoch = maxEpoche;
    }

    public AlphaModifierType getAlphaModifierType() {
        return alphaModifierType;
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

    @Override
    public String toString() {
        return "{" +
                "layers=" + Arrays.toString(layers) +
                ", maxAlpha=" + maxAlpha +
                ", minAlpha=" + minAlpha +
                ", alphaModType=" + alphaModifierType +
                ", maxEpoche=" + maxEpoch +
                '}';
    }

    enum AlphaModifierType {
        LINEAR,
        EXP
    }
}