package org.rsultan.core.ensemble.isolationforest.evaluation;

import static java.util.stream.IntStream.range;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.rsultan.core.Evaluator;
import org.rsultan.core.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class TPRThresholdEvaluator implements Evaluator<Double, IsolationForest> {

  private static final Logger LOG = LoggerFactory.getLogger(TPRThresholdEvaluator.class);
  private double desiredTPR = 0.9;
  private double learningRate = 0.01;
  private double TPR = 0;
  private double FPR = 0;

  private double trainTestThreshold = 0.5D;
  private int responseVariableIndex = -1;
  private Dataframe testDataframe;

  public Double evaluate(IsolationForest trainable, Dataframe dataframe) {
    var dfSplit = dataframe.copy().shuffle().trainTest(trainTestThreshold);

    final INDArray train = dfSplit[0];
    final INDArray test = testDataframe == null ? dfSplit[1] : testDataframe.toMatrix();

    if (responseVariableIndex < 0) {
      responseVariableIndex = test.columns() - 1;
    }

    final int[] predictorIndices = range(0, train.columns()).filter(i -> i != responseVariableIndex)
        .toArray();
    var trained = trainable.setUseAnomalyScoresOnly(true)
        .train(train.getColumns(predictorIndices));

    var testDf = test.getColumns(predictorIndices);
    var responses = test.getColumn(responseVariableIndex);

    var predict = trained.predict(testDf.getColumns(predictorIndices));

    double threshold = 1;
    while (threshold > 0 && TPR <= desiredTPR) {
      LOG.info("Evaluating isolation forest with threshold {}", threshold);
      threshold -= learningRate;
      var scores = Nd4j.create(predict.toDoubleVector());
      BooleanIndexing.replaceWhere(scores, 1.0, Conditions.greaterThanOrEqual(threshold));
      BooleanIndexing.replaceWhere(scores, 0.0, Conditions.lessThan(threshold));
      double truePositives = 0, trueNegatives = 0, falsePositives = 0, falseNegative = 0;

      for (int i = 0; i < predict.rows(); i++) {
        var response = responses.getLong(i, 0);
        var prediction = scores.getLong(i, 0);
        truePositives += response == 1L && prediction == 1L ? 1L : 0L;
        trueNegatives += response == 0L && prediction == 0L ? 1L : 0L;
        falsePositives += response == 0L && prediction == 1L ? 1L : 0L;
        falseNegative += response == 1L && prediction == 0L ? 1L : 0L;
      }
      TPR = truePositives / (truePositives + falseNegative);
      TPR = Double.isNaN(TPR) ? 0 : TPR;

      FPR = falsePositives / (falsePositives + trueNegatives);
      FPR = Double.isNaN(FPR) ? 0L : FPR;
    }

    if (threshold < 0) {
      throw new IllegalArgumentException("Cannot have desired TPR");
    }

    return threshold;
  }

  public TPRThresholdEvaluator setDesiredTPR(double desiredTPR) {
    this.desiredTPR = desiredTPR;
    return this;
  }

  public TPRThresholdEvaluator setLearningRate(double learningRate) {
    this.learningRate = learningRate;
    return this;
  }

  public TPRThresholdEvaluator setResponseVariableIndex(int responseVariableIndex) {
    this.responseVariableIndex = responseVariableIndex;
    return this;
  }

  public TPRThresholdEvaluator setTrainTestThreshold(double trainTestThreshold) {
    this.trainTestThreshold = trainTestThreshold;
    return this;
  }

  public TPRThresholdEvaluator setTestDataframe(Dataframe testDataframe) {
    this.testDataframe = testDataframe;
    return this;
  }

  public void showMetrics() {
    Dataframes.create(new String[]{"TPR", "FPR"}, List.of(List.of(TPR, FPR))).show(10);
  }
}
