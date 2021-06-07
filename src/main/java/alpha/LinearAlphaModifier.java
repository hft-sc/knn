package alpha;

public class LinearAlphaModifier implements AlphaModifier {

    private final int maxEpoch;
    private final double maxAlpha;
    private final double minAlpha;

    public LinearAlphaModifier(int maxEpoch, double maxAlpha, double minAlpha) {
        this.maxEpoch = maxEpoch;
        this.maxAlpha = maxAlpha;
        this.minAlpha = minAlpha;
    }

    @Override
    public double modify(int currentEpoche, double currentAlpha) {
        return currentAlpha - (maxAlpha - minAlpha) / maxEpoch;
    }
}
