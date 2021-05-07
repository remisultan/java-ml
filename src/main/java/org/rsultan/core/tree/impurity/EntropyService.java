package org.rsultan.core.tree.impurity;

import static java.util.Arrays.stream;
import static java.util.Map.Entry.comparingByKey;
import static java.util.Map.entry;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.summingDouble;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.LongStream.range;
import static org.nd4j.linalg.indexing.BooleanIndexing.replaceWhere;
import static org.nd4j.linalg.indexing.conditions.Conditions.isInfinite;
import static org.nd4j.linalg.ops.transforms.Transforms.log;

import java.util.Map;
import java.util.Map.Entry;
import java.util.function.LongFunction;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class EntropyService implements ImpurityService {

  private final int totalLabels;

  public EntropyService(int totalLabels) {
    this.totalLabels = totalLabels;
  }

  @Override
  public INDArray compute(INDArray labels) {
    var countVectors = range(0, labels.columns()).parallel()
        .mapToObj(getCountsPerVector(labels))
        .map(doubleArray -> Nd4j.create(doubleArray, doubleArray.length, 1))
        .collect(toList());
    var countMatrix = Nd4j.create(countVectors, countVectors.size(), countVectors.get(0).rows());
    var sumPerColumn = countMatrix.sum(true, 1);
    var probabilities = countMatrix.div(sumPerColumn);
    var logProb = log(probabilities, 2, true);
    replaceWhere(logProb, 0.0, isInfinite());
    return probabilities.mul(logProb).sum(true, 1).transpose().neg();
  }

  private LongFunction<double[]> getCountsPerVector(INDArray labels) {
    var map = getEmptyCountMap();
    return idx -> {
      var counts = getExistingLabelCounts(labels, idx);
      map.forEach(counts::putIfAbsent);
      return getSortedCountArray(counts);
    };
  }

  private Map<Double, Double> getEmptyCountMap() {
    return IntStream.range(0, totalLabels).mapToDouble(i -> i).boxed()
        .collect(toMap(i -> i, i -> 0D));
  }

  private Map<Double, Double> getExistingLabelCounts(INDArray labels, long idx) {
    return stream(labels.getColumn(idx).toDoubleVector())
        .mapToObj(value -> entry(value, 1))
        .collect(groupingBy(Entry::getKey, summingDouble(e -> 1D)));
  }

  private double[] getSortedCountArray(Map<Double, Double> counts) {
    return counts.entrySet().stream()
        .sorted(comparingByKey())
        .map(Entry::getValue)
        .mapToDouble(Double::valueOf)
        .toArray();
  }
}
