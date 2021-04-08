package org.rsultan.example;

import static java.awt.image.BufferedImage.TYPE_INT_RGB;
import static java.lang.Double.parseDouble;
import static java.lang.Integer.parseInt;
import static java.lang.System.currentTimeMillis;
import static java.util.stream.IntStream.range;

import java.awt.Color;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import javax.imageio.ImageIO;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.kmedoids.KMeans;
import org.rsultan.core.clustering.kmedoids.KMedians;
import org.rsultan.core.clustering.kmedoids.evaluation.KMedoidEvaluator;
import org.rsultan.core.clustering.kmedoids.strategy.InitialisationStrategy;
import org.rsultan.core.clustering.type.MedoidType;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;

public class KMedoidExample {

  /*
   Make sure args[0] /path/to/your/image.(jpg|png)
   Make sure args[1] K >= 1
   Make sure args[2] MEAN|MEDIAN
   Make sure args[3] RANDOM|PLUS_PLUS
   Make sure args[4] Epochs
   Make sure args[5] image factor
   Make sure args[6] directory
   Make sure args[7] filename prefix
  */
  static {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
  }

  public static void main(String[] args)
      throws IOException, ExecutionException, InterruptedException {

    int K = parseInt(args[1]);
    var algorithm = MedoidType.valueOf(args[2]);
    var initialisation = InitialisationStrategy.valueOf(args[3]);
    int epochs = parseInt(args[4]);

    var kMedoid = switch (algorithm) {
      case MEAN -> new KMeans(K, epochs, initialisation);
      case MEDIAN -> new KMedians(K, epochs, initialisation);
    };

    var originalImg = ImageIO.read(new File(args[0]));
    var imgFactor = parseDouble(args[5]);
    int width = (int) (originalImg.getWidth() * imgFactor);
    int height = (int) (originalImg.getHeight() * imgFactor);
    var img = resizeImage(originalImg, width, height);

    var df = getDataframeFromImage(img);
    System.out.println("Dataframe loaded");

    System.out.println("Starting...");
    var start = currentTimeMillis();
    kMedoid.train(df);
    System.out.println(kMedoid + "took " + ((currentTimeMillis() - start) / 1000) + " seconds");
    kMedoid.showMetrics();

    var outputImage = new BufferedImage(width, height, TYPE_INT_RGB);
    var squaredCluster = Nd4j.create(kMedoid.getCluster().toDoubleVector(), height, width);

    range(0, width).parallel().unordered().forEach(x ->
        range(0, height).parallel().unordered().forEach(y -> {
          var rgb1 = kMedoid.getCentroids().getRow(squaredCluster.getLong(y, x));
          int color = new Color(rgb1.getInt(0), rgb1.getInt(1), rgb1.getInt(2)).getRGB();
          outputImage.setRGB(x, y, color);
        })
    );

    var directory = args[6];
    var outputPrefix = args[7];
    var fileName = directory + kMedoid.getK() + "_" + algorithm + "_" + outputPrefix + ".png";
    ImageIO.write(outputImage, "png", new File(fileName));
  }

  private static Dataframe getDataframeFromImage(BufferedImage img) {
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
    return Dataframes.create(red, green, blue);
  }

  private static BufferedImage resizeImage(BufferedImage originalImage, int targetWidth,
      int targetHeight) {
    var resultingImage = originalImage
        .getScaledInstance(targetWidth, targetHeight, Image.SCALE_SMOOTH);
    var outputImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
    outputImage.getGraphics().drawImage(resultingImage, 0, 0, null);
    return outputImage;
  }
}
