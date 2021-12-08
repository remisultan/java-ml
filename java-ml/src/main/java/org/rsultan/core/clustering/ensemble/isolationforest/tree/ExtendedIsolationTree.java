package org.rsultan.core.clustering.ensemble.isolationforest.tree;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.nd4j.common.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.RawTrainable;
import org.rsultan.core.clustering.ensemble.domain.IsolationNode;
import org.rsultan.core.clustering.ensemble.isolationforest.tree.ExtendedIsolationTree.Slope;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import static java.util.stream.LongStream.range;

public class ExtendedIsolationTree extends AbstractTree<Slope> implements RawTrainable<ExtendedIsolationTree> {

  public static final Logger LOG = LoggerFactory.getLogger(ExtendedIsolationTree.class);
  private final int extensionLevel;

  public ExtendedIsolationTree(int treeDepthLimit, int extensionLevel) {
    super(treeDepthLimit);
    this.extensionLevel = extensionLevel;
  }

  @Override
  public ExtendedIsolationTree train(INDArray matrix) {
    this.tree = buildTree(matrix, treeDepthLimit);
    return this;
  }

  @Override
  protected IsolationNode<Slope> buildNode(INDArray X, int currentDepth) {
    LOG.info("Tree Depth {}", currentDepth);
    if (currentDepth <= 0 || X.rows() <= 2) {
      return new IsolationNode<>(X);
    }
    int numberOfFeatures = X.columns();
    var mins = X.min(true, 0);
    var maxs = X.max(true, 0);
    var n = getNormalVector(mins.shape());
    var p = getIntercept(numberOfFeatures, mins, maxs);
    var w = X.sub(p).mmul(n);

    var left = getVector(X, getIndices(w, idx -> w.getDouble(idx) < 0));
    var right = getVector(X, getIndices(w, idx -> w.getDouble(idx) >= 0));

    return new IsolationNode<>(
        new Slope(n, p),
        buildTree(left, currentDepth - 1),
        buildTree(right, currentDepth - 1)
    );
  }

  private INDArray getIntercept(int numberOfFeatures, INDArray mins, INDArray maxs) {
    var p = Nd4j.zeros(mins.shape());
    for (int i = 0; i < numberOfFeatures; i++) {
      p.put(0, i, MathUtils.randomDoubleBetween(mins.getDouble(0, i), maxs.getDouble(0, i)));
    }
    return p;
  }

  private INDArray getNormalVector(long[] shape) {
    var distribution = new NormalDistribution();
    var n = Nd4j.zeros(shape);
    for (int i = 0; i < n.columns(); i++) {
      n.put(0, i, distribution.sample());
    }
    final int numSamples = n.columns() - this.extensionLevel - 1;
    if (numSamples > 0) {
      final INDArray arange = Nd4j.create(range(0, n.columns()).boxed().toList());
      int[] indices = Nd4j.choice(arange,  Nd4j.rand(n.columns()), numSamples).toIntVector();
      for (int index : indices) {
        n.putScalar(index, 0.0D);
      }
    }
    return n.transpose();
  }

  @Override
  protected boolean chooseLeftNode(INDArray row, Slope slope) {
    return row.sub(slope.p).mmul(slope.n).getDouble(0) < 0;
  }

  static record Slope(INDArray n, INDArray p){}

}
