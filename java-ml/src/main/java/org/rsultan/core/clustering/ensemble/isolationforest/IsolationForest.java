package org.rsultan.core.clustering.ensemble.isolationforest;

import static java.util.stream.IntStream.range;
import static org.rsultan.core.clustering.ensemble.isolationforest.utils.ScoreUtils.averagePathLength;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.LongStream;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.core.RawTrainable;
import org.rsultan.core.Trainable;
import org.rsultan.core.clustering.ensemble.isolationforest.tree.IsolationTree;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IsolationForest implements Trainable<IsolationForest> {

  private static final Logger LOG = LoggerFactory.getLogger(IsolationTree.class);
  protected final int nbTrees;
  protected double anomalyThreshold = 0.5;
  protected List<? extends RawTrainable<?>> isolationTrees;
  protected int sampleSize = 256;
  private boolean useAnomalyScoresOnly;

  public IsolationForest(int nbTrees) {
    this.nbTrees = nbTrees;
  }

  public IsolationForest setSampleSize(int sampleSize) {
    this.sampleSize = sampleSize;
    return this;
  }

  public IsolationForest setAnomalyThreshold(double anomalyThreshold) {
    this.anomalyThreshold = anomalyThreshold;
    return this;
  }

  public IsolationForest setUseAnomalyScoresOnly(boolean useAnomalyScoresOnly) {
    this.useAnomalyScoresOnly = useAnomalyScoresOnly;
    return this;
  }

  @Override
  public IsolationForest train(Dataframe dataframe) {
    var matrix = dataframe.toMatrix();
    int realSample = sampleSize >= matrix.rows() ? sampleSize / 10 : sampleSize;
    int treeDepth = (int) Math.ceil(Math.log(realSample) / Math.log(2));
    isolationTrees = range(0, nbTrees).parallel()
        .peek(i -> LOG.info("Tree number: {}", i))
        .mapToObj(i -> LongStream.range(0, realSample)
            .map(idx -> RandomUtils.nextLong(0, matrix.rows()))
            .toArray())
        .map(NDArrayIndex::indices)
        .map(matrix::get)
        .map(m -> getTree(treeDepth, m))
        .toList();
    return this;
  }

  protected RawTrainable<?> getTree(int treeDepth, INDArray m) {
    return new IsolationTree(treeDepth).train(m);
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var matrix = dataframe.toMatrix();
    var anomalyScores = computeAnomalyScore(matrix);
    final DoubleStream doubleStream = DoubleStream.of(
            anomalyScores.toDoubleVector()
    );
    var isAnomaly =  new Column<>("anomalies",
            useAnomalyScoresOnly ? doubleStream.boxed().toArray() : doubleStream.mapToObj(score -> score >= anomalyThreshold ? 1L : 0L).toArray());
    return dataframe.addColumn(isAnomaly);
  }

  protected INDArray computeAnomalyScore(INDArray matrix) {
    var pathLengths = new ArrayList<INDArray>();
    for (RawTrainable<?> tree: isolationTrees) {
      LOG.info("Compute paths for tree {}", isolationTrees.indexOf(tree) + 1);
      pathLengths.add(tree.predict(matrix));
    }
    int[] shape = {pathLengths.size(), pathLengths.get(0).columns()};
    var avgLength = Nd4j.create(pathLengths, shape).mean(true, 0);
    var twos = Nd4j.ones(avgLength.shape()).mul(2D);
    return Transforms.pow(twos, avgLength.neg().div(averagePathLength(sampleSize)));
  }
}
