package org.rsultan.core.tree.impurity;

import static java.util.Arrays.stream;
import static java.util.function.Function.identity;
import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;

import java.util.Map;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class AbstractImpurityService implements ImpurityService {

  @Override
  public Map<Double, Long> getClassCount(INDArray labels) {
    return stream(labels.toDoubleVector()).boxed().collect(groupingBy(identity(), counting()));
  }


  protected INDArray computeProbabilities(INDArray classCount) {
    var sumPerColumn = classCount.sum(true, 1);
    return classCount.div(sumPerColumn);
  }

}
