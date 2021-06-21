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
        var sb = new StringBuilder();


        sb.append(String.format("%50s", parameters.toString())).append('\t');
      //  sb.append("Genauigkeit:").append(String.format(Locale.ENGLISH, "%,.4f", result[21])).append('\t');
          sb.append("Trefferquote Richtig erkannt:").append(String.format(Locale.ENGLISH, "%,.4f",  result[21])).append('\t');
          sb.append("Trefferqoute Falsch erkannt:").append(String.format(Locale.ENGLISH, "%,.4f", result[22])).append('\t');
     //   sb.append("richtigPositiv:").append(String.format("%4d", (int) result[6])).append('\t');//    sb.append("falschNegativ:").append(String.format("%4d", (int) result[9])).append('\t');
     //   sb.append("richtigNegativ:").append(String.format("%4d", (int) result[8])).append('\t');
    //    sb.append("falschPositiv:").append(String.format("%4d", (int) result[7]));

        System.out.println(sb);
    }

}
