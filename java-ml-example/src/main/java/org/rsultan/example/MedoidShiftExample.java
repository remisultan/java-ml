package org.rsultan.example;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import javax.imageio.ImageIO;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.medoidshift.MeanShift;
import org.rsultan.core.clustering.medoidshift.MedianShift;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;

public class MedoidShiftExample {

  /*
   Make sure args[0] /path/to/your/image.(jpg|png)
  */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args)
      throws IOException, ExecutionException, InterruptedException {

    var img = ImageIO.read(new File(args[0]));

    List<List<?>> list = new ArrayList<>();
    for (int y = 0; y < img.getHeight(); y++) {
      for (int x = 0; x < img.getWidth(); x++) {
        int pixel = img.getRGB(x, y);
        var color = new Color(pixel, true);
        list.add(List.of(
            color.getRed(),
            color.getGreen(),
            color.getBlue()
        ));
      }
    }

    String[] columns = {"r", "g", "b"};
    var df = Dataframes.create(columns, list);

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
    System.out.println("MeanShift Centroids : " + meanShift.getCentroids());

    var trainedMedianShift = futureMedianShift.get();
    System.out.println("MedianShift Centroids : " + medianShift.getCentroids());

  }
}
