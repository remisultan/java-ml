package org.rsultan.example;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import javax.imageio.ImageIO;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.kmedoids.KMeans;
import org.rsultan.core.clustering.kmedoids.KMedians;
import org.rsultan.core.clustering.kmedoids.KMedoids;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;

public class KMedoidExample {

  /*
   The data used is the infamous IRIS dataset
   Make sure args[0] /path/to/your/src/main/resources/softmax/iris.data
  */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args)
      throws IOException, ExecutionException, InterruptedException {

    var img = ImageIO.read(new File(args[0]));

    var red = new Column<Integer>("r", new ArrayList<>());
    var green = new Column<Integer>("g", new ArrayList<>());
    var blue = new Column<Integer>("b", new ArrayList<>());


    for (int y = 0; y < img.getHeight(); y++) {
      for (int x = 0; x < img.getWidth(); x++) {
        int pixel = img.getRGB(x, y);
        var color = new Color(pixel, true);
        red.values().add(color.getRed());
        green.values().add(color.getGreen());
        blue.values().add(color.getBlue());
      }
    }

    var df = Dataframes.create(red, green, blue);

    System.out.println("Dataframe loaded");

    final KMedoids kMeans = new KMeans(5, 10);
    final KMedoids kMedians = new KMedians(5, 10);

    var futureKmeans = CompletableFuture.supplyAsync(() -> {
      var start = System.currentTimeMillis();
      var train = kMeans.train(df);
      System.out.println("Kmeans took:" + (System.currentTimeMillis() - start) / 1000D);
      return train;
    });

    var futureKmedian = CompletableFuture.supplyAsync(() -> {
      var start = System.currentTimeMillis();
      var train = kMedians.train(df);
      System.out.println("Kmedians took:" + (System.currentTimeMillis() - start) / 1000D);
      return train;
    });

    var trainedkMeans = futureKmeans.get();
    kMeans.showMetrics();
    System.out.println("KMeans Error: " + kMeans.getLoss());

    var trainedkMedians = futureKmedian.get();
    kMedians.showMetrics();
    System.out.println("KMedians Error: " + kMedians.getLoss());

    var img1 = new BufferedImage(img.getWidth(), img.getHeight(),
        BufferedImage.TYPE_INT_RGB);
    var img2 = new BufferedImage(img.getWidth(), img.getHeight(),
        BufferedImage.TYPE_INT_RGB);

    var squaredKmeansCluster = Nd4j.create(trainedkMeans.getCluster().toDoubleVector(), img.getHeight(), img.getWidth());
    var squaredKmediansCluster = Nd4j.create(trainedkMedians.getCluster().toDoubleVector(), img.getHeight(), img.getWidth());

    for (int y = 0; y < img.getHeight(); y++) {
      for (int x = 0; x < img.getWidth(); x++) {
        var rgb1 = trainedkMeans.getC().getRow(squaredKmeansCluster.getLong(y, x));
        var rgb2 = trainedkMedians.getC().getRow(squaredKmediansCluster.getLong(y, x));
        img1.setRGB(x, y, new Color(rgb1.getInt(0), rgb1.getInt(1), rgb1.getInt(2)).getRGB());
        img2.setRGB(x, y, new Color(rgb2.getInt(0), rgb2.getInt(1), rgb2.getInt(2)).getRGB());
      }
    }
    ImageIO.write(img1, "png", new File(args[1] + "kmeans.png"));
    ImageIO.write(img2, "png", new File(args[1] + "kmedians.png"));
  }
}
