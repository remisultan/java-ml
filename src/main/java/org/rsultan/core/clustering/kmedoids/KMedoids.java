package org.rsultan.core.clustering.kmedoids;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static java.util.stream.LongStream.rangeClosed;
import static org.apache.commons.lang3.RandomUtils.nextLong;

import java.util.Arrays;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.rsultan.core.clustering.Clustering;
import org.rsultan.core.clustering.kmedoids.centroid.MedoidFactory;
import org.rsultan.core.clustering.kmedoids.type.KMedoidType;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class KMedoids implements Clustering {

  private final MedoidFactory kMedoidFactory;
  private final int K;
  private final int numberOfIterations;

  private INDArray C;
  private INDArray D;
  private INDArray X;
  private INDArray Xt;
  private double loss = -1;
  private INDArray cluster;

  public KMedoids(int k, int numberOfIterations, KMedoidType kMedoidType) {
    this.K = k;
    this.numberOfIterations = numberOfIterations;
    this.kMedoidFactory = kMedoidType.getMedoidFactory();
  }

  @Override
  public KMedoids train(Dataframe dataframe) {
    X = dataframe.toMatrix();
    Xt = X.transpose();
    C = Nd4j.create(range(0, K)
        .map(k -> nextLong(0, Xt.columns()))
        .mapToObj(Xt::getColumn)
        .collect(toList()), K, Xt.rows());

    range(0, numberOfIterations)
        .filter(epoch -> loss != 0)
        .forEach(epoch -> {
          D = computeDistance(kMedoidFactory);
          cluster = Nd4j.argMin(D, 1);

          var newMedoids = range(0, K).parallel()
              .mapToObj(k -> {
                long[] longs = range(0, Xt.columns()).parallel()
                    .filter(xCol -> k == cluster.getLong(xCol))
                    .toArray();
                INDArrayIndex[] indices = {NDArrayIndex.all(), new SpecifiedIndex(longs)};
                return Xt.get(indices).transpose();
              }).map(kMedoidFactory::computeMedoids)
              .collect(toList());

          var newCenters = Nd4j.create(newMedoids, K, Xt.rows());
          loss = kMedoidFactory.computeNorm(C.sub(newCenters));
          C = newCenters;
        });

    return this;
  }

  public void showMetrics() {
    var centroids = range(0, C.rows()).boxed()
        .map(idx -> Arrays.toString(C.getRow(idx).toDoubleVector()))
        .collect(toList());
    var indices = new Column<>("", rangeClosed(1, K).boxed().collect(toList()));
    Dataframes.create(indices, new Column<>("centroids", centroids)).tail();
  }

  protected INDArray computeDistance(MedoidFactory medoidFactory) {
    return medoidFactory.computeDistance(C, X).transpose();
  }

  public INDArray getC() {
    return C;
  }

  public double getLoss() {
    return loss;
  }

  public int getK() {
    return K;
  }

  public INDArray getCluster() {
    return cluster;
  }
}
