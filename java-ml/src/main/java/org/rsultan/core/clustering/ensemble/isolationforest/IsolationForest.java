package org.rsultan.core.clustering.ensemble.isolationforest;

import static java.util.stream.IntStream.range;
import static org.rsultan.core.clustering.ensemble.isolationforest.utils.ScoreUtils.averagePathLength;

import java.util.List;
import java.util.stream.DoubleStream;
import org.apache.commons.lang3.RandomUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.core.Trainable;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IsolationForest implements Trainable<IsolationForest> {

  private static final Logger LOG = LoggerFactory.getLogger(IsolationTree.class);
  private final int nbTrees;
  private final double anomalyThreshold;
  private List<IsolationTree> isolationTrees;
  private int sampleSize = 256;

  public IsolationForest(int nbTrees, double anomalyThreshold) {
    this.nbTrees = nbTrees;
    this.anomalyThreshold = anomalyThreshold;
  }

  public IsolationForest setSampleSize(int sampleSize) {
    this.sampleSize = sampleSize;
    return this;
  }

  @Override
  public IsolationForest train(Dataframe dataframe) {
    var matrix = dataframe.toMatrix();
    int realSample = sampleSize >= matrix.rows() ? sampleSize / 10 : sampleSize;
    int treeDepth = (int) Math.ceil(Math.log(realSample) / Math.log(2));
    isolationTrees = range(0, nbTrees)
        .peek(i -> LOG.info("Tree number: {}", i))
        .mapToObj(i -> range(0, realSample)
            .map(idx -> RandomUtils.nextInt(0, matrix.rows()))
            .toArray()).map(matrix::getRows)
        .map(new IsolationTree(treeDepth)::train)
        .toList();
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var matrix = dataframe.toMatrix();
    var anomalyScores = computeAnomalyScore(matrix);
    var isAnomaly = new Column<>("anomalies", DoubleStream.of(
        anomalyScores.toDoubleVector()
    ).mapToObj(score -> score >= anomalyThreshold).toArray());
    return dataframe.addColumn(isAnomaly);
  }

  private INDArray computeAnomalyScore(INDArray matrix) {
    var pathLengths = isolationTrees.stream().parallel()
        .peek(tree -> LOG.info("Compute paths for tree {}", isolationTrees.indexOf(tree)))
        .map(tree -> tree.predict(matrix)).toList();
    int[] shape = {pathLengths.size(), pathLengths.get(0).columns()};
    var avgLength = Nd4j.create(pathLengths, shape).mean(true, 0);
    var twos = Nd4j.ones(avgLength.shape()).mul(2D);
    return Transforms.pow(twos, avgLength.neg().div(averagePathLength(sampleSize)));
  }
}
