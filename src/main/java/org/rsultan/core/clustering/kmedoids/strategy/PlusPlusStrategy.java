package org.rsultan.core.clustering.kmedoids.strategy;

import static org.apache.commons.lang3.RandomUtils.nextDouble;
import static org.apache.commons.lang3.RandomUtils.nextInt;
import static org.nd4j.linalg.indexing.conditions.Conditions.greaterThanOrEqual;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;

import java.util.ArrayList;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.centroid.MedoidFactory;

public class PlusPlusStrategy implements InitialisationFactory {

  private final MedoidFactory medoidFactory;

  public PlusPlusStrategy(MedoidFactory medoidFactory) {
    this.medoidFactory = medoidFactory;
  }

  @Override
  public INDArray initialiseCenters(long K, INDArray X) {
    var Xt = X.transpose();
    var centers = new ArrayList<INDArray>();
    addNewCenters(X, centers, nextInt(0, X.columns()));
    var C = Nd4j.empty();
    while (centers.size() <= K) {
      C = Nd4j.create(centers, centers.size(), X.rows());
      var squaredDistances = Nd4j.min(pow(medoidFactory.computeDistance(C, Xt), 2), 0);
      var probabilities = squaredDistances.div(squaredDistances.sum(true, 0));
      var cumulativeSumTheshold = probabilities.cumsum(0)
          .getWhere(nextDouble(0, 1), greaterThanOrEqual());
      addNewCenters(X, centers, X.columns() - cumulativeSumTheshold.columns() + 1);
    }
    return C;
  }

  private void addNewCenters(INDArray X, ArrayList<INDArray> centers, int index) {
    centers.add(X.getColumn(index));
  }
}
