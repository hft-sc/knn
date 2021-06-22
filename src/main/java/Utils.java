import java.util.Locale;

public class Utils {

    public static void printResult(double[] result) {
        System.out.println("Anzahl Muster:  \t" + result[0]);
        System.out.println("Anzahl Positiv: \t" + result[1]);
        System.out.println("Anzahl Negativ: \t" + result[2]);
        System.out.println("Anteil Positiv: \t" + result[3]);
        System.out.println("Anteil Negativ: \t" + result[4]);
        System.out.println("Genauigkeit  :  \t" + result[5]);
        System.out.println("Trefferquote:   \t" + result[10]);
        System.out.println("Ausfallrate :   \t" + result[11]);
        System.out.println("richtigPositiv: \t" + result[6]);
        System.out.println("falsch Negativ: \t" + result[9]);
        System.out.println("richtigNegativ: \t" + result[8]);
        System.out.println("falsch Positiv: \t" + result[7]);
    }

    public static void printResult(TestParameters parameters, double[] result) {


        String sb = String.format("%50s", parameters.toString()) + '\t' +
                "richtig+:" + String.format(Locale.ENGLISH, "%,.4f", result[21]) + '\t' +
                "falsch-:" + String.format(Locale.ENGLISH, "%,.4f", result[22]) + '\t' +
                "0 +:" + (int) result[1] + "  " +
                "0 -:" + (int) result[11] + "  " +
                "1 +:" + (int) result[2] + "  " +
                "1 -:" + (int) result[12] + "  " +
                "2 +:" + (int) result[3] + "  " +
                "2 -:" + (int) result[13] + "  " +
                "3 +:" + (int) result[4] + "  " +
                "3 -:" + (int) result[14] + "  " +
                "4 +:" + (int) result[5] + "  " +
                "4 -:" + (int) result[15] + "  " +
                "5 +:" + (int) result[6] + "  " +
                "5 -:" + (int) result[16] + "  " +
                "6 +:" + (int) result[7] + "  " +
                "6 -:" + (int) result[17] + "  " +
                "7 +:" + (int) result[8] + "  " +
                "7 -:" + (int) result[18] + "  " +
                "8 +:" + (int) result[9] + "  " +
                "8 -:" + (int) result[19] + "  " +
                "9 +:" + (int) result[10] + "  " +
                "9 -:" + (int) result[20] + "  ";
        System.out.println(sb);
    }

}
