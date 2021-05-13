package org.rsultan.core.tree.impurity;

import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

import java.util.stream.IntStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;

public abstract class AbstractImpurityService implements ImpurityService {

  protected final int totalLabels;

  protected AbstractImpurityService(int totalLabels) {
    this.totalLabels = totalLabels;
  }

  @Override
  public INDArray getClassCount(INDArray labels) {
    var classCount = IntStream.range(0, labels.columns()).parallel()
        .mapToObj(labels::getColumn)
        .map(label -> range(0, totalLabels).parallel()
            .mapToObj(i -> label.getWhere((double) i, Conditions.equals()))
            .mapToDouble(vector -> vector == null ? 0.0D : vector.columns())
            .toArray())
        .map(doubleArray -> Nd4j.create(doubleArray, doubleArray.length, 1))
        .collect(toList());
    return Nd4j.create(classCount, classCount.size(), classCount.get(0).rows());
  }


  protected INDArray computeProbabilities(INDArray classCount) {
    var sumPerColumn = classCount.sum(true, 1);
    return classCount.div(sumPerColumn);
  }

}
