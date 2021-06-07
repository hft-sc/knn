package alpha;

public class ExpAlphaModifier implements AlphaModifier {

    private static final int minX = 0;
    private static final int maxX = 10;

    private final int maxEpoch;
    private final double maxAlpha;
    private final double minAlpha;

    public ExpAlphaModifier(int maxEpoch, double maxAlpha, double minAlpha) {
        this.maxEpoch = maxEpoch;
        this.maxAlpha = maxAlpha;
        this.minAlpha = minAlpha;
    }

    @Override
    public double modify(int currentEpoch, double currentAlpha) {
        double progress = 1.0 * currentEpoch / maxEpoch;
        double x = progress * (maxX - minX);
        return Math.exp(-x) * (maxAlpha - minAlpha) + minAlpha;
    }
}
