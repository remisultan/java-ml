package org.rsultan.core.clustering.kmedoids;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.apache.commons.lang3.RandomUtils.nextLong;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.Clustering;
import org.rsultan.core.clustering.kmedoids.centroid.MedoidFactory;
import org.rsultan.core.clustering.kmedoids.type.KMedoidType;
import org.rsultan.dataframe.Dataframe;

public abstract class AbstractKMedoid implements Clustering {

  private final KMedoidType kMedoidType;
  private final int K;
  private final int numberOfIterations;

  private INDArray C;
  private INDArray D;
  private INDArray X;
  private double error = -1;

  public AbstractKMedoid(int k, int numberOfIterations, KMedoidType kMedoidType) {
    this.K = k;
    this.numberOfIterations = numberOfIterations;
    this.kMedoidType = kMedoidType;
  }

  @Override
  public AbstractKMedoid train(Dataframe dataframe) {
    X = dataframe.toMatrix().transpose();
    C = Nd4j.create(range(0, K)
        .map(k -> nextLong(0, X.columns())).boxed()
        .map(X::getColumn)
        .collect(toList()), K, X.rows());
    var medoidFactory = this.kMedoidType.getMedoidFactory();
    range(0, numberOfIterations)
        .filter(epoch -> error != 0)
        .forEach(epoch -> {
          D = computeDistance(medoidFactory);
          var cluster = Nd4j.argMin(D, 1);
          var newMeans = range(0, K).boxed().map(k -> range(0, X.columns())
              .filter(xCol -> k.equals(cluster.getLong(xCol)))
              .boxed()
              .map(idx -> X.getColumn(idx)).collect(toList()))
              .map(cols -> Nd4j.create(cols, cols.size(), X.rows()))
              .map(medoidFactory::computeMedoids)
              .collect(toList());
          var newCenters = Nd4j.create(newMeans, K, X.rows());
          error = C.sub(newCenters).norm1Number().doubleValue();
          C = newCenters;
        });

    return this;
  }

  protected INDArray computeDistance(MedoidFactory medoidFactory) {
    return Nd4j.create(
        range(0, X.columns()).boxed().map(X::getColumn)
            .map(column -> medoidFactory.computeDistance(C, column))
            .collect(toList()), X.columns(), K);
  }

  public INDArray getC() {
    return C;
  }

  public double getError() {
    return error;
  }
}
