package org.rsultan.example;

import static java.awt.image.BufferedImage.TYPE_INT_RGB;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.kmedoids.KMeans;
import org.rsultan.core.clustering.kmedoids.KMedians;
import org.rsultan.core.clustering.medoidshift.MeanShift;
import org.rsultan.core.clustering.medoidshift.MedianShift;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;

public class MedoidShiftExample {

  /*
   Make sure args[0] /path/to/your/image.(jpg|png)
   Make sure args[1] /output/directory/path/
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

    final MeanShift meanShift = new MeanShift(50, 100);
    final MedianShift medianShift = new MedianShift(70, 100);

    var futureMeanShift = CompletableFuture.supplyAsync(() -> {
      var start = System.currentTimeMillis();
      var train = meanShift.train(df);
      System.out.println("MeanShift took:" + (System.currentTimeMillis() - start) / 1000D);
      return train;
    });

    var futureMedianShift = CompletableFuture.supplyAsync(() -> {
      var start = System.currentTimeMillis();
      var train = medianShift.train(df);
      System.out.println("MedianShift took:" + (System.currentTimeMillis() - start) / 1000D);
      return train;
    });

    var trainedMeanShift = futureMeanShift.get();
    System.out.println("MeanShift Centroids : " + meanShift.getC());

    var trainedMedianShift = futureMedianShift.get();
    System.out.println("MedianShift Centroids : " + medianShift.getC());

  }
}
