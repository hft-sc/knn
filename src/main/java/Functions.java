public class Functions {

    public static double sigmuid(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    public static double sigmuidDerivative(double x) {
        final var value = sigmuid(x);
        return (value * (1 - value));
    }
}
