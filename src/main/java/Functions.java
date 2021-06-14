import org.jblas.DoubleMatrix;

public class Functions {

    public static double sigmoid(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    public static DoubleMatrix sigmoid(DoubleMatrix matrix) {
        var result = new DoubleMatrix(matrix.rows, matrix.columns);
        for (int i = 0; i < matrix.length; i++) {
            result.put(i, sigmoid(matrix.get(i)));
        }
        return result;
    }

    public static double sigmoidDerivative(double x) {
        final var value = sigmoid(x);
        return (value * (1 - value));
    }

    public static DoubleMatrix sigmoidDerivative(DoubleMatrix matrix) {
        var result = new DoubleMatrix(matrix.rows, matrix.columns);
        for (int i = 0; i < matrix.length; i++) {
            result.put(i, sigmoidDerivative(matrix.get(i)));
        }
        return result;
    }
}
