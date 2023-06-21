package org.rsultan.core.evaluation;

import static java.util.Comparator.comparingDouble;
import static java.util.stream.IntStream.range;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.nd4j.common.primitives.Quad;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.rsultan.core.Evaluator;
import org.rsultan.core.RawTrainable;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class AreaUnderCurve<T extends RawTrainable<T>> implements Evaluator<AreaUnderCurve<T>, T> {

  private static final Logger LOG = LoggerFactory.getLogger(AreaUnderCurve.class);
  private double learningRate = 0.01;
  private double trainTestThreshold = 0.5D;
  private int responseVariableIndex = -1;
  private Dataframe testDataframe;

  private Map<Double, ConfusionMatrix> rocData;
  private double f1Score;
  private double phiCoef;

  public AreaUnderCurve<T> evaluate(T trainable, Dataframe dataframe) {
    rocData = new TreeMap<>();
    var dfSplit = dataframe.copy().shuffle().trainTest(trainTestThreshold);

    final INDArray train = dfSplit[0];
    final INDArray test = testDataframe == null ? dfSplit[1] : testDataframe.toMatrix();

    if (responseVariableIndex < 0) {
      responseVariableIndex = test.columns() - 1;
    }

    final int[] predictorIndices = range(0, train.columns()).filter(i -> i != responseVariableIndex)
        .toArray();
    var trained = trainable.train(train.getColumns(predictorIndices));

    var testDf = test.getColumns(predictorIndices);
    var responses = test.getColumn(responseVariableIndex);

    var predict = trained.predict(testDf.getColumns(predictorIndices));

    double threshold = 1;
    while (threshold > 0) {
      LOG.info("Evaluating isolation forest with threshold {}", threshold);
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
      var cm = new ConfusionMatrix(truePositives, trueNegatives, falsePositives, falseNegative);
      rocData.put(threshold, cm);
      final double lastF1Score = cm.f1Score();
      final double lastPhiCoef = cm.phiCoefficient();
      if(f1Score <= lastF1Score){
        f1Score = lastF1Score;
        phiCoef = lastPhiCoef;
        LOG.info("F1 score for threshold: {} --> {}", threshold, lastF1Score);
        LOG.info("Phi coefficient for threshold: {} --> {}", threshold, lastPhiCoef);
      }
      threshold -= learningRate;
    }

    return this;
  }

  public double getAUC() {
    var sortedDots = rocData.values().stream()
        .map(cf -> Map.entry(cf.fallout(), cf.recall()))
        .sorted(Entry.comparingByKey()).toList();

    double auc = 0D;
    for (int i = 0; i < sortedDots.size() - 1; i++) {
      var e1 = sortedDots.get(i);
      var e2 = sortedDots.get(i + 1);

      double base = e2.getKey() - e1.getKey();
      double diffHeight = e2.getValue() - e1.getValue();
      double minHeight = diffHeight < 0 ? e2.getValue() : e1.getValue();
      auc += (minHeight * base) + (diffHeight * base / 2);
    }
    return auc;
  }

  public AreaUnderCurve<T> setLearningRate(double learningRate) {
    this.learningRate = learningRate;
    return this;
  }

  public AreaUnderCurve<T> setResponseVariableIndex(int responseVariableIndex) {
    this.responseVariableIndex = responseVariableIndex;
    return this;
  }

  public AreaUnderCurve<T> setTrainTestThreshold(double trainTestThreshold) {
    this.trainTestThreshold = trainTestThreshold;
    return this;
  }

  public AreaUnderCurve<T> setTestDataframe(Dataframe testDataframe) {
    this.testDataframe = testDataframe;
    return this;
  }
}
