import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;
import java.awt.*;

public class GUI {

    public static void zeichnen(double[][] daten, double[][] flaeche) {

        JFrame f = new JFrame();
        JLabel label = new JLabel("Loading ...");
        label.setBounds(260, 250, 80, 40);
        f.add(label);
        f.setSize(600, 600);
        f.setLayout(null);
        f.setVisible(true);
        JFreeChart chart;

        chart = testcreateChart(daten, flaeche);
        ChartPanel panel = new ChartPanel(chart);
        panel.setPreferredSize(f.getSize());
        f.setContentPane(panel);
        SwingUtilities.updateComponentTreeUI(f);

    }

    private static JFreeChart testcreateChart(double[][] daten, double[][] flaeche) {


        XYSeries hyperplane = areaSeries(flaeche, "Flaeche");
        XYSeries class1points = pointSeries(daten, 0, "KlasseNegativ");
        XYSeries class2points = pointSeries(daten, 1, "KlassePositiv");

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(class1points);
        dataset.addSeries(class2points);
        dataset.addSeries(hyperplane);

        JFreeChart chart = ChartFactory.createXYLineChart("", "", "", dataset, PlotOrientation.VERTICAL, false, false, false);


        XYPlot plot = (XYPlot) chart.getPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        //HOMchart.getPlot().setBackgroundPaint( Color.WHITE );


        renderer.setSeriesLinesVisible(1, false);
        renderer.setSeriesShapesVisible(1, true);

        renderer.setSeriesPaint(1, Color.red);
        renderer.setSeriesLinesVisible(2, false);
        renderer.setSeriesShapesVisible(2, true);

        renderer.setSeriesPaint(2, Color.yellow);
        //HOMrenderer.setSeriesPaint(2, Color.GRAY);

//	    Shape s = renderer.getSeriesShape(1);
        renderer.setSeriesLinesVisible(0, false);
        renderer.setSeriesShapesVisible(0, true);
        renderer.setSeriesPaint(0, Color.blue);
        plot.setRenderer(renderer);


        return chart;
    }


    private static XYSeries areaSeries(double[][] flaeche, String key) {
        XYSeries series = new XYSeries(key);
        for (int i = 0; i < flaeche.length; i++) {
            double wert1 = flaeche[i][0];
            double wert2 = flaeche[i][1];
            series.add(wert1, wert2);
        }
        return series;
    }

    private static XYSeries pointSeries(double[][] vectors, int y, String key) {
        XYSeries series = new XYSeries(key);
        for (int i = 0; i < vectors.length; i++) {
            if ((int) vectors[i][2] == y) {
                series.add(100. * vectors[i][0], 100 * vectors[i][1]);//es wird wieder angenommen, dass die Eingabewerte durch 100 geteilt wurden
            }
        }
        return series;
    }
}
