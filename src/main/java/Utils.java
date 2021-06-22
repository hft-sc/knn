import java.util.Locale;

public class Utils {

    public static void printParameters(TestParameters parameters) {
        System.out.print(String.format("%50s", parameters.toString()) + '\t');
    }

    public static void printResult(double[] result) {
        String sb = "richtig+:" + String.format(Locale.ENGLISH, "%,.4f", result[21]) + '\t' +
                "falsch-:" + String.format(Locale.ENGLISH, "%,.4f", result[22]) + '\t' +
                "0+: " + (int) result[1] + "  " +
                "0-: " + (int) result[11] + "  " +
                "1+: " + (int) result[2] + "  " +
                "1-: " + (int) result[12] + "  " +
                "2+: " + (int) result[3] + "  " +
                "2-: " + (int) result[13] + "  " +
                "3+: " + (int) result[4] + "  " +
                "3-: " + (int) result[14] + "  " +
                "4+: " + (int) result[5] + "  " +
                "4-: " + (int) result[15] + "  " +
                "5+: " + (int) result[6] + "  " +
                "5-: " + (int) result[16] + "  " +
                "6+: " + (int) result[7] + "  " +
                "6-: " + (int) result[17] + "  " +
                "7+: " + (int) result[8] + "  " +
                "7-: " + (int) result[18] + "  " +
                "8+: " + (int) result[9] + "  " +
                "8-: " + (int) result[19] + "  " +
                "9+: " + (int) result[10] + "  " +
                "9-: " + (int) result[20] + "  ";
        System.out.println(sb);
    }
}
