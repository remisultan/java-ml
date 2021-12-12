package org.rsultan.core.clustering.ensemble.isolationforest;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.RawTrainable;
import org.rsultan.core.clustering.ensemble.isolationforest.tree.ExtendedIsolationTree;

public class ExtendedIsolationForest extends IsolationForest {

  private final int extensionLevel;

  public ExtendedIsolationForest(int nbTrees, int extensionLevel) {
    super(nbTrees);
    this.extensionLevel = extensionLevel;
  }

  public ExtendedIsolationForest setSampleSize(int sampleSize) {
    super.setSampleSize(sampleSize);
    return this;
  }

  public ExtendedIsolationForest setAnomalyThreshold(double anomalyThreshold) {
    super.setAnomalyThreshold(anomalyThreshold);
    return this;
  }

  public ExtendedIsolationForest setUseAnomalyScoresOnly(boolean useAnomalyScoresOnly) {
    super.setUseAnomalyScoresOnly(useAnomalyScoresOnly);
    return this;
  }

  @Override
  protected RawTrainable<?> getTree(int treeDepth, INDArray m) {
    return new ExtendedIsolationTree(treeDepth, extensionLevel).train(m);
  }
}
