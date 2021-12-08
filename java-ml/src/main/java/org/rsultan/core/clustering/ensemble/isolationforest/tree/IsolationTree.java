package org.rsultan.core.clustering.ensemble.isolationforest.tree;

import static org.apache.commons.lang3.RandomUtils.nextInt;

import org.nd4j.common.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.RawTrainable;
import org.rsultan.core.clustering.ensemble.domain.IsolationNode;
import org.rsultan.core.clustering.ensemble.isolationforest.tree.IsolationTree.Feature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IsolationTree extends AbstractTree<Feature> implements RawTrainable<IsolationTree> {

  public static final Logger LOG = LoggerFactory.getLogger(IsolationTree.class);

  public IsolationTree(int treeDepthLimit) {
    super(treeDepthLimit);
  }

  @Override
  public IsolationTree train(INDArray matrix) {
    this.tree = buildTree(matrix, treeDepthLimit);
    return this;
  }

  @Override
  protected IsolationNode<Feature> buildNode(INDArray X, int currentDepth) {
    LOG.info("Tree Depth {}", currentDepth);
    int numberOfFeatures = X.columns();
    int splitFeature = nextInt(0, numberOfFeatures);
    var feature = X.getColumn(splitFeature);
    double startInclusive = feature.minNumber().doubleValue();
    double endInclusive = feature.maxNumber().doubleValue();
    double valueSplit = MathUtils.randomDoubleBetween(startInclusive, endInclusive);

    var left = getVector(X, getIndices(feature, idx -> feature.getDouble(idx) < valueSplit));
    var right = getVector(X, getIndices(feature, idx -> feature.getDouble(idx) >= valueSplit));

    return new IsolationNode<>(
        new Feature(splitFeature, valueSplit),
        buildTree(left, currentDepth - 1),
        buildTree(right, currentDepth - 1)
    );
  }

  @Override
  protected boolean chooseLeftNode(INDArray row, Feature feature) {
    return row.getDouble(feature.feature) < feature.threshold ;
  }

  static record Feature(int feature, double threshold){}

}
