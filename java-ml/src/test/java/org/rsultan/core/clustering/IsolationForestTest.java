package org.rsultan.core.clustering;

import static java.util.stream.IntStream.rangeClosed;
import static org.apache.commons.lang3.RandomUtils.nextDouble;
import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.rsultan.core.evaluation.AreaUnderCurve;
import org.rsultan.core.ensemble.isolationforest.ExtendedIsolationForest;
import org.rsultan.core.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;

public class IsolationForestTest {

  private final static List<List<?>> rows = Stream.of(0, 3, 6, 7, 8, 100)
      .map(IsolationForestTest::createCircleDataAroundCenter)
      .flatMap(List::stream)
      .collect(Collectors.toList());

  private static List<List<Double>> createCircleDataAroundCenter(int radius) {
    return rangeClosed(-radius, radius).mapToDouble(x -> x).boxed().flatMap(x -> {
      double y = Math.sqrt(radius * radius - x * x);
      double response = nextDouble() > 0.7 ? 1L : 0L;
      return Stream.of(List.of(x, y, response), List.of(x, -y, response));
    }).toList();
  }

  @Test
  public void must_perform_isolation() {
    var df = Dataframes.create(new String[]{"x", "y", "response"}, rows);
    final double anomalyThreshold = 0.4;
    var predict = new IsolationForest(10).setAnomalyThreshold(anomalyThreshold)
        .setUseAnomalyScoresOnly(true)
        .setSampleSize(15)
        .train(df).predict(df);

    int all = predict.copy().getResult().rows().size();
    int anomalies = predict.copy().filter("anomalies", (Double d) -> d >= anomalyThreshold)
        .getResult().rows().size();
    int nonAnomalies = predict.copy().filter("anomalies", (Double d) -> d < anomalyThreshold)
        .getResult().rows().size();

    predict.copy().show(all);

    assertThat(anomalies).isBetween(1, all);
    assertThat(nonAnomalies).isEqualTo(all - anomalies);
  }

  @Test
  public void must_perform_isolation_without_anomaly_scores() {
    var df = Dataframes.create(new String[]{"x", "y", "response"}, rows);
    final double anomalyThreshold = 0.4;
    var predict = new IsolationForest(10).setAnomalyThreshold(anomalyThreshold)
        .setUseAnomalyScoresOnly(false)
        .setSampleSize(15)
        .train(df).predict(df);
    int all = predict.copy().getResult().rows().size();
    int anomalies = predict.copy().filter("anomalies", (Double d) -> d == 1)
        .getResult()
        .rows().size();
    int nonAnomalies = predict.copy().filter("anomalies", (Double d) -> d == 0)
        .getResult().rows().size();

    assertThat(anomalies).isBetween(1, all);
    assertThat(nonAnomalies).isEqualTo(all - anomalies);
  }

  @Test
  public void should_evaluate_tpr() {
    var df = Dataframes.create(new String[]{"x", "y", "response"}, rows);
    var model = new IsolationForest(10).setSampleSize(15).setUseAnomalyScoresOnly(true);
    var evaluator = new AreaUnderCurve<IsolationForest>()
        .setResponseVariableIndex(2)
        .setTrainTestThreshold(0.75)
        .setLearningRate(0.01)
        .evaluate(model, df);

    assertThat(evaluator.getAUC()).isBetween(0.0, 1.0);
  }

  @Test
  public void must_perform_extended_isolation_forest() {
    var df = Dataframes.create(new String[]{"x", "y", "response"}, rows);
    final double anomalyThreshold = 0.4;
    var predict = new ExtendedIsolationForest(10, 1).setAnomalyThreshold(anomalyThreshold)
        .setUseAnomalyScoresOnly(true)
        .setSampleSize(15)
        .train(df).predict(df);

    int all = predict.copy().getResult().rows().size();
    int anomalies = predict.copy().filter("anomalies", (Double d) -> d >= anomalyThreshold)
        .getResult().rows().size();
    int nonAnomalies = predict.copy().filter("anomalies", (Double d) -> d < anomalyThreshold)
        .getResult().rows().size();

    predict.copy().show(all);

    assertThat(anomalies).isBetween(1, all);
    assertThat(nonAnomalies).isEqualTo(all - anomalies);
  }

  @Test
  public void must_perform_extended_isolation_without_anomaly_scores() {
    var df = Dataframes.create(new String[]{"x", "y", "response"}, rows);
    final double anomalyThreshold = 0.4;
    var predict = new ExtendedIsolationForest(10, 1).setAnomalyThreshold(anomalyThreshold)
        .setUseAnomalyScoresOnly(false)
        .setSampleSize(15)
        .train(df).predict(df);
    int all = predict.copy().getResult().rows().size();
    int anomalies = predict.copy().filter("anomalies", (Double d) -> d == 1)
        .getResult()
        .rows().size();
    int nonAnomalies = predict.copy().filter("anomalies", (Double d) -> d == 0)
        .getResult().rows().size();

    assertThat(anomalies).isBetween(1, all);
    assertThat(nonAnomalies).isEqualTo(all - anomalies);
  }

  @Test
  public void should_evaluate_tpr_with_extended() {
    var df = Dataframes.create(new String[]{"x", "y", "response"}, rows);
    var model = new ExtendedIsolationForest(10, 1).setUseAnomalyScoresOnly(true).setSampleSize(15);
    var evaluator = new AreaUnderCurve<IsolationForest>()
        .setResponseVariableIndex(2)
        .setTrainTestThreshold(0.75)
        .setLearningRate(0.01)
        .evaluate(model, df);

    assertThat(evaluator.getAUC()).isBetween(0.0, 1.0);
  }
}
