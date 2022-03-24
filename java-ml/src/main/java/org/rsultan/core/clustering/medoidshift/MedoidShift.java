package org.rsultan.core.clustering.medoidshift;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static java.util.stream.LongStream.rangeClosed;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.Clustering;
import org.rsultan.core.clustering.type.MedoidType;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class MedoidShift implements Clustering {

  private static final Logger LOG = LoggerFactory.getLogger(MedoidShift.class);

  private final MedoidType medoidType;
  private final double bandwidth;
  private final long epoch;

  private INDArray centroids;
  private INDArray Xt;

  protected MedoidShift(double bandwidth, long epoch, MedoidType medoidType) {
    this.medoidType = medoidType;
    this.epoch = epoch;
    this.bandwidth = bandwidth;
  }

  @Override
  public MedoidShift train(Dataframe dataframe) {
    var medoidFactory = medoidType.getMedoidFactory();
    var isTerminated = new AtomicBoolean(false);
    Xt = dataframe.toMatrix().transpose();
    centroids = Xt.dup();
    range(0, epoch)
        .filter(epoch -> !isTerminated.get())
        .forEach(epoch -> {
          LOG.info("Epoch : {}", epoch);
          var newCentroidList = range(0, centroids.columns())
              .parallel().unordered()
              .mapToObj(col -> {
                var centroid = centroids.getColumn(col);
                var range = epoch == 0 ? range(col, Xt.columns()) : range(0, Xt.columns());
                return range.parallel().unordered().mapToObj(Xt::getColumn)
                    .filter(feature -> medoidFactory.computeNorm(feature.sub(centroid)) < bandwidth)
                    .collect(toList());
              })
              .map(features -> Nd4j.create(features, features.size(), centroids.rows()))
              .map(medoidFactory::computeMedoids)
              .distinct()
              .collect(toList());
          var newC = Nd4j.create(newCentroidList, newCentroidList.size(), centroids.rows())
              .transpose();
          if (centroids.equalShapes(newC) && centroids.equals(newC) || centroids.columns() == 1) {
            isTerminated.set(true);
          }
          centroids = newC;
        });

    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var medoidFactory = medoidType.getMedoidFactory();
    var Xpredict = dataframe.toMatrix();
    var distances = medoidFactory.computeDistance(Xpredict, centroids.transpose());
    var indices = Nd4j.argMin(distances, 1);
    var predictions = centroids.transpose().get(indices);
    return dataframe.copy().addColumn("predictions",
        range(0, predictions.rows()).mapToObj(predictions::getRow).collect(toList()));
  }


  public void showMetrics() {
    var centroids = range(0, this.centroids.columns()).boxed()
        .map(idx -> Arrays.toString(this.centroids.getColumn(idx).toDoubleVector()))
        .collect(toList());
    final List<List<?>> collect = rangeClosed(1, this.centroids.columns()).boxed().map(List::of)
        .collect(toList());
    Dataframes.create(new String[]{"K"}, collect)
        .addColumn("centroids", centroids)
        .show(centroids.size());
  }

  public INDArray getCentroids() {
    return centroids;
  }
}
