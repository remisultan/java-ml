package org.rsultan.core.clustering.ensemble.evaluation;

import org.rsultan.core.Evaluator;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.TrainTestDataframe;

public class TPRThresholdEvaluator implements Evaluator<Double, IsolationForest> {

  private final String responseVariable;
  private final String predictionColumn;
  private double desiredTPR = 0.9;
  private double learningRate = 0.01;
  private double TPR = 0;
  private double FPR = 0;

  public TPRThresholdEvaluator(String responseVariable, String predictionColumn) {
    this.responseVariable = responseVariable;
    this.predictionColumn = predictionColumn;
  }

  @Override
  public Double evaluate(IsolationForest trainable, TrainTestDataframe dataframe) {
    var dfSplit = dataframe.shuffle().split();
    double threshold = 1;
    while (threshold > 0 && TPR <= desiredTPR) {
      threshold -= learningRate;
      var trained = trainable.setAnomalyThreshold(threshold)
          .train(dfSplit.train().mapWithout(responseVariable));
      var responses = dfSplit.test().<Long>get(responseVariable);
      var predictions = trained.predict(dfSplit.test().mapWithout(responseVariable))
          .<Long>get(predictionColumn);

      double truePositives = 0;
      double trueNegatives = 0;
      double falsePositives = 0;
      double falseNegative = 0;

      for (int i = 0; i < responses.size(); i++) {
        var response = responses.get(i);
        var prediction = predictions.get(i);
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

  public void showMetrics() {
    Dataframes.create(new String[]{"TPR", "FPR"}, new Row(TPR, FPR)).tail();
  }
}
