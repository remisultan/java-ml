package org.rsultan.core.clustering.ensemble.isolationforest;

import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.rsultan.core.clustering.ensemble.isolationforest.tree.ExtendedIsolationTree;
import org.rsultan.dataframe.Dataframe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.LongStream;

import static java.lang.String.format;
import static java.util.stream.IntStream.range;
import static org.rsultan.core.clustering.ensemble.isolationforest.utils.ScoreUtils.averagePathLength;

public class ExtendedIsolationForest extends IsolationForest {

  private static final Logger LOG = LoggerFactory.getLogger(ExtendedIsolationForest.class);

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

  @Override
  public ExtendedIsolationForest train(Dataframe dataframe) {
    var matrix = dataframe.toMatrix();
    if (extensionLevel > matrix.columns() - 1 || extensionLevel < 0) {
      throw new IllegalArgumentException(format("extensionLevel must be between 0 and %d, current is [%d]", matrix.columns() - 1, extensionLevel));
    }
    int realSample = sampleSize >= matrix.rows() ? sampleSize / 10 : sampleSize;
    int treeDepth = (int) Math.ceil(Math.log(realSample) / Math.log(2));
    isolationTrees = range(0, nbTrees).parallel()
        .peek(i -> LOG.info("Tree number: {}", i))
        .mapToObj(i -> LongStream.range(0, realSample)
                .map(idx -> RandomUtils.nextLong(0, matrix.rows()))
                .toArray())
        .map(NDArrayIndex::indices)
        .map(matrix::get)
        .map(m -> new ExtendedIsolationTree(treeDepth, extensionLevel).train(m))
        .toList();
    return this;
  }
}
