package org.rsultan.core.clustering.ensemble.isolationforest;

import static java.util.stream.IntStream.range;
import static org.apache.commons.lang3.RandomUtils.nextDouble;
import static org.apache.commons.lang3.RandomUtils.nextInt;
import static org.rsultan.core.clustering.ensemble.isolationforest.utils.ScoreUtils.averagePathLength;

import java.util.stream.LongStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.RawTrainable;
import org.rsultan.core.clustering.ensemble.domain.IsolationNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IsolationTree implements RawTrainable<IsolationTree> {

  public static final Logger LOG = LoggerFactory.getLogger(IsolationTree.class);
  private final int treeDepthLimit;
  private IsolationNode tree;

  public IsolationTree(int treeDepthLimit) {
    this.treeDepthLimit = treeDepthLimit;
  }

  @Override
  public IsolationTree train(INDArray matrix) {
    this.tree = buildTree(matrix, treeDepthLimit);
    return this;
  }

  private IsolationNode buildTree(INDArray matrix, int currentDepth) {
    LOG.info("Tree Depth {}", currentDepth);
    if (currentDepth <= 0 || matrix.rows() <= 2) {
      return new IsolationNode(matrix);
    }
    int numberOfFeatures = matrix.columns();
    int splitFeature = nextInt(0, numberOfFeatures);
    var feature = matrix.getColumn(splitFeature);
    double startInclusive = feature.minNumber().doubleValue();
    double endInclusive = feature.maxNumber().doubleValue();

    double valueSplit =
        getValueSplit(startInclusive, endInclusive);

    var leftIndices = range(0, feature.columns()).parallel()
        .filter(idx -> feature.getDouble(idx) < valueSplit)
        .toArray();
    var left = matrix.getRows(leftIndices);

    var rightIndices = range(0, feature.columns()).parallel()
        .filter(idx -> feature.getDouble(idx) > valueSplit)
        .toArray();
    var right = matrix.getRows(rightIndices);

    return new IsolationNode(
        splitFeature,
        valueSplit,
        buildTree(left, currentDepth - 1),
        buildTree(right, currentDepth - 1)
    );
  }

  private double getValueSplit(double startInclusive, double endInclusive) {
    if (startInclusive < 0 && endInclusive < 0) {
      return -nextDouble(endInclusive * -1, startInclusive * -1);
    } else if (startInclusive < 0 && endInclusive >= 0) {
      return nextDouble(0, endInclusive + startInclusive * -1) + startInclusive;
    }
    return nextDouble(startInclusive, endInclusive);
  }

  @Override
  public INDArray predict(INDArray matrix) {
    return Nd4j.create(LongStream.range(0, matrix.rows()).boxed()
        .map(matrix::getRow)
        .map(row -> {
          var node = tree;
          int length = 0;
          while (!node.isLeaf()) {
            if (row.getDouble(node.feature()) < node.featureThreshold()) {
              node = node.left();
            } else {
              node = node.right();
            }
            length++;
          }
          int leafSize = node.data().rows();
          return length + averagePathLength(leafSize);
        }).toList());
  }
}
