package org.rsultan.core.clustering.kmedoids.evaluation;

import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.rangeClosed;
import static org.rsultan.core.clustering.type.MedoidType.MEAN;

import java.util.ArrayList;
import java.util.List;
import org.rsultan.core.clustering.kmedoids.KMeans;
import org.rsultan.core.clustering.kmedoids.KMedians;
import org.rsultan.core.clustering.type.MedoidType;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;

public class KMedoidEvaluator {

  public static final String SUM_OF_SQUARES = "sumOfSquares";
  public static final String K = "K";
  private final int minK;
  private final int maxK;
  private final MedoidType medoidType;
  private final int epoch;
  private final List<Integer> clusters;

  public KMedoidEvaluator(int minK, int maxK, MedoidType medoidType) {
    this(minK, maxK, medoidType, 100);
  }

  public KMedoidEvaluator(int minK, int maxK, MedoidType medoidType, int epoch) {
    this.minK = minK;
    this.maxK = maxK;
    this.medoidType = medoidType;
    this.epoch = epoch;
    clusters = rangeClosed(minK, maxK).boxed().collect(toList());
  }

  public Dataframe evaluate(Dataframe dataframe) {
    var values = new ArrayList<List<?>>(clusters.size());
    rangeClosed(minK, maxK).parallel()
        .mapToObj(k -> switch (medoidType) {
              case MEDIAN -> new KMedians(k, epoch);
              case MEAN -> new KMeans(k, epoch);
            }
        ).map(kMedoids -> kMedoids.train(dataframe.copy()))
        .forEachOrdered(kMedoids -> {
          var indexOfK = clusters.indexOf(kMedoids.getK());
          values.add(indexOfK, List.of(kMedoids.getK(), kMedoids.getWCSS()));
        });
    return Dataframes.create(new String[]{K, SUM_OF_SQUARES}, values);
  }

  @Override
  public String toString() {
    return "KMedoidEvaluator{" +
        "minK=" + minK +
        ", maxK=" + maxK +
        ", medoidType=" + medoidType +
        ", epoch=" + epoch +
        '}';
  }
}
