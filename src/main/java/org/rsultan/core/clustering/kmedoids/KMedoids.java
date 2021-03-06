package org.rsultan.core.clustering.kmedoids;

import static java.util.Optional.ofNullable;
import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static java.util.stream.LongStream.rangeClosed;

import java.util.Arrays;
import java.util.stream.LongStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.rsultan.core.clustering.Clustering;
import org.rsultan.core.clustering.centroid.MedoidFactory;
import org.rsultan.core.clustering.kmedoids.strategy.InitialisationStrategy;
import org.rsultan.core.clustering.type.MedoidType;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class KMedoids implements Clustering {

  private static final Logger LOG = LoggerFactory.getLogger(KMedoids.class);

  protected final MedoidType medoidType;
  protected final int K;
  protected final int numberOfIterations;
  protected final InitialisationStrategy initialisationStrategy;

  private INDArray centroids;
  private INDArray distances;
  private INDArray X;
  private INDArray Xt;
  private Double loss;
  private INDArray cluster;
  private INDArray WCSS;

  protected KMedoids(
      int k,
      int numberOfIterations,
      MedoidType kMedoidType,
      InitialisationStrategy initialisationStrategy) {
    this.K = k;
    this.numberOfIterations = numberOfIterations;
    this.medoidType = kMedoidType;
    this.initialisationStrategy = initialisationStrategy;
  }

  @Override
  public KMedoids train(Dataframe dataframe) {
    X = dataframe.toMatrix();
    Xt = X.transpose();
    var medoidFactory = medoidType.getMedoidFactory();

    LOG.info("Initialising centroids");
    centroids = buildInitialCentroids(medoidFactory);
    LOG.info("Centroids initialised");

    range(0, numberOfIterations)
        .filter(epoch -> ofNullable(loss).orElse(-1.0D) != 0.0D)
        .forEach(epoch -> {
          LOG.info("Epoch {} {}", epoch, medoidType);
          distances = medoidFactory.computeDistance(centroids, X).transpose();
          cluster = Nd4j.argMin(distances, 1);

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
          loss = medoidFactory.computeNorm(centroids.sub(newCenters));
          centroids = newCenters;
          LOG.info("Epoch {}, Loss : {} for {}", epoch, loss, medoidType);
        });
    this.WCSS = distances.transpose().mmul(distances).sum().div(distances.rows());
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var medoidFactory = medoidType.getMedoidFactory();
    var Xpredict = dataframe.toMatrix();
    var distances = medoidFactory.computeDistance(centroids, Xpredict).transpose();
    var centers = LongStream.of(Nd4j.argMin(distances, 1).toLongVector()).boxed().collect(toList());
    return dataframe.addColumn(new Column<>("K", centers))
        .<Long, INDArray>map("prediction", centroids::getRow, "K");
  }

  private INDArray buildInitialCentroids(MedoidFactory medoidFactory) {
    return this.initialisationStrategy.initialiseCenters(K, Xt, medoidFactory);
  }

  public void showMetrics() {
    var centroids = range(0, this.centroids.rows()).boxed()
        .map(idx -> Arrays.toString(this.centroids.getRow(idx).toDoubleVector()))
        .collect(toList());
    var indices = new Column<>("K", rangeClosed(1, K).boxed().collect(toList()));
    Dataframes.create(indices, new Column<>("centroids", centroids)).tail();
  }

  public INDArray getCentroids() {
    return centroids;
  }

  public double getLoss() {
    return loss;
  }

  public int getK() {
    return K;
  }

  public INDArray getWCSS() {
    return WCSS;
  }

  public INDArray getCluster() {
    return cluster;
  }
}
