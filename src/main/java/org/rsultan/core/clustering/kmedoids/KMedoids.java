package org.rsultan.core.clustering.kmedoids;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static java.util.stream.LongStream.rangeClosed;
import static org.apache.commons.lang3.RandomUtils.nextLong;

import java.util.Arrays;
import java.util.stream.LongStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.rsultan.core.clustering.Clustering;
import org.rsultan.core.clustering.centroid.MedoidFactory;
import org.rsultan.core.clustering.type.MedoidType;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class KMedoids implements Clustering {

  private static final Logger LOG = LoggerFactory.getLogger(KMedoids.class);

  private final MedoidType medoidType;
  private final int K;
  private final int numberOfIterations;

  private INDArray C;
  private INDArray D;
  private INDArray X;
  private INDArray Xt;
  private double loss = -1;
  private INDArray cluster;

  public KMedoids(int k, int numberOfIterations, MedoidType kMedoidType) {
    this.K = k;
    this.numberOfIterations = numberOfIterations;
    this.medoidType = kMedoidType;
  }

  @Override
  public KMedoids train(Dataframe dataframe) {
    X = dataframe.toMatrix();
    Xt = X.transpose();
    C = buildInitialCentroids();
    var medoidFactory = medoidType.getMedoidFactory();

    range(0, numberOfIterations)
        .filter(epoch -> loss != 0)
        .forEach(epoch -> {
          LOG.info("Epoch {}, Loss : {} for {}", epoch, loss, medoidType);
          D = medoidFactory.computeDistance(C, X).transpose();
          cluster = Nd4j.argMin(D, 1);

          var newMedoids = range(0, K).parallel().unordered()
              .mapToObj(k -> {
                long[] indicesArray = range(0, Xt.columns()).parallel().unordered()
                    .filter(xCol -> k == cluster.getLong(xCol))
                    .toArray();
                INDArrayIndex[] indices = {NDArrayIndex.all(), new SpecifiedIndex(indicesArray)};
                INDArray matrix = Xt.get(indices).transpose();
                return matrix.isEmpty() ? Nd4j.zeros(X.shape()) : matrix;
              }).map(medoidFactory::computeMedoids)
              .collect(toList());

          var newCenters = Nd4j.create(newMedoids, K, Xt.rows());
          loss = medoidFactory.computeNorm(C.sub(newCenters));
          C = newCenters;
        });

    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var medoidFactory = medoidType.getMedoidFactory();
    var Xpredict = dataframe.toMatrix();
    var distances =  medoidFactory.computeDistance(C, Xpredict).transpose();
    var centers = LongStream.of(Nd4j.argMin(distances, 1).toLongVector()).boxed().collect(toList());
    return dataframe.addColumn(new Column<>("K", centers))
        .<Long, INDArray>withColumn("prediction", "K", C::getRow);
  }

  private INDArray buildInitialCentroids() {
    return Nd4j.create(range(0, K)
        .map(k -> nextLong(0, Xt.columns()))
        .mapToObj(Xt::getColumn)
        .collect(toList()), K, Xt.rows());
  }

  public void showMetrics() {
    var centroids = range(0, C.rows()).boxed()
        .map(idx -> Arrays.toString(C.getRow(idx).toDoubleVector()))
        .collect(toList());
    var indices = new Column<>("K", rangeClosed(1, K).boxed().collect(toList()));
    Dataframes.create(indices, new Column<>("centroids", centroids)).tail();
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
