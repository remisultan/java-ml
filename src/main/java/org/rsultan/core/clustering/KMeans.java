package org.rsultan.core.clustering;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.apache.commons.lang3.RandomUtils.nextLong;
import static org.nd4j.linalg.ops.transforms.Transforms.allEuclideanDistances;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframe;

public class KMeans implements Clustering {

  private final int K;
  private final int numberOfIterations;
  private INDArray C;
  private INDArray D;
  private INDArray X;
  private double error = -1;

  public KMeans(int k, int numberOfIterations) {
    this.K = k;
    this.numberOfIterations = numberOfIterations;
  }

  @Override
  public KMeans train(Dataframe dataframe) {
    X = dataframe.toMatrix().transpose();
    C = Nd4j.create(range(0, K)
        .map(k -> nextLong(0, X.columns())).boxed()
        .map(X::getColumn)
        .collect(toList()), K, X.rows());

    range(0, numberOfIterations)
        .filter(epoch -> error != 0)
        .forEach(epoch -> {
          D = computeDistance();
          var cluster = Nd4j.argMin(D, 1);
          var newMeans = range(0, K).boxed().map(k -> range(0, X.columns())
              .filter(xCol -> k.equals(cluster.getLong(xCol)))
              .boxed()
              .map(idx -> X.getColumn(idx)).collect(toList())
          ).map(cols -> Nd4j.create(cols, cols.size(), X.rows()).mean(true, 0))
              .collect(toList());
          var newCenters = Nd4j.create(newMeans, K, X.rows());
          error = C.sub(newCenters).norm1Number().doubleValue();
          C = newCenters;
        });

    return this;
  }

  private INDArray computeDistance() {
    return Nd4j.create(
        range(0, X.columns()).boxed().map(X::getColumn)
            .map(column -> allEuclideanDistances(C, column, 1))
            .collect(toList()), X.columns(), K);
  }

  public INDArray getC() {
    return C;
  }

  public double getError() {
    return error;
  }
}
