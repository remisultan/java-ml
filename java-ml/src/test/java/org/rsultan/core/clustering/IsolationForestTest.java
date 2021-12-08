package org.rsultan.core.clustering;

import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.rangeClosed;
import static org.apache.commons.lang3.RandomUtils.nextBoolean;
import static org.apache.commons.lang3.RandomUtils.nextDouble;
import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.rsultan.core.clustering.ensemble.evaluation.TPRThresholdEvaluator;
import org.rsultan.core.clustering.ensemble.isolationforest.ExtendedIsolationForest;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.Row;

public class IsolationForestTest {

  private static Row[] rows;

  static {
    rows = List.of(0, 3, 6, 7, 8).stream()
        .map(IsolationForestTest::createCircleDataAroundCenter)
        .flatMap(List::stream)
        .toArray(Row[]::new);
  }

  private static List<Row> createCircleDataAroundCenter(int radius) {
    return rangeClosed(-radius, radius).boxed().flatMap(x -> {
      double y = Math.sqrt(radius * radius - x * x);
      long response = nextDouble() > 0.7 ? 1L : 0L;
      return List.of(new Row(x, y, response), new Row(x, -y, response)).stream();
    }).collect(toList());
  }

  @Test
  public void must_perform_isolation() {
    var df = Dataframes.create(new String[]{"x", "y", "response"}, rows);
    var predict = new IsolationForest(10).setAnomalyThreshold(0.56).setSampleSize(15).train(df).predict(df);
    int anomalies = predict.filter("anomalies", obj -> obj.equals(1L)).getRowSize();
    int nonAnomalies = predict.filter("anomalies", obj -> obj.equals(0L)).getRowSize();

    assertThat(anomalies).isBetween(1, predict.getRowSize());
    assertThat(nonAnomalies).isEqualTo(predict.getRowSize() - anomalies);
  }

  @Test
  public void should_evaluate_tpr() {
    var df = Dataframes.trainTest(new String[]{"x", "y", "response"}, rows);
    var model = new IsolationForest(10).setSampleSize(15);
    var evaluator = new TPRThresholdEvaluator("response", "anomalies").setDesiredTPR(0.9).setLearningRate(0.01);
    var threshold = evaluator.evaluate(model, df);
    evaluator.showMetrics();
    assertThat(threshold).isGreaterThan(0.4);
  }

  @Test
  public void must_perform_extended_isolation() {
    var df = Dataframes.create(new String[]{"x", "y", "response"}, rows);
    var predict = new ExtendedIsolationForest(10, 1).setAnomalyThreshold(0.56).setSampleSize(15).train(df).predict(df);
    int anomalies = predict.filter("anomalies", obj -> obj.equals(1L)).getRowSize();
    int nonAnomalies = predict.filter("anomalies", obj -> obj.equals(0L)).getRowSize();

    assertThat(anomalies).isBetween(1, predict.getRowSize());
    assertThat(nonAnomalies).isEqualTo(predict.getRowSize() - anomalies);
  }

  @Test
  public void should_evaluate_extended_tpr() {
    var df = Dataframes.trainTest(new String[]{"x", "y", "response"}, rows);
    var model = new ExtendedIsolationForest(10, 1).setSampleSize(15);
    var evaluator = new TPRThresholdEvaluator("response", "anomalies").setDesiredTPR(0.9).setLearningRate(0.01);
    var threshold = evaluator.evaluate(model, df);
    evaluator.showMetrics();
    assertThat(threshold).isGreaterThan(0.4);
  }

  @Test
  public void must_perform_isolation_and_retrieve_scores() {
    var df = Dataframes.create(new String[]{"x", "y", "response"}, rows);
    var predict = new IsolationForest(10).setAnomalyThreshold(0.56).setSampleSize(15).setUseAnomalyScoresOnly(true).train(df).predict(df);
    int anomalies = predict.filter("anomalies", (Number obj) -> obj.doubleValue() > 0.56).getRowSize();
    int nonAnomalies = predict.filter("anomalies", (Number obj) -> obj.doubleValue() <= 0.56).getRowSize();

    assertThat(anomalies).isBetween(1, predict.getRowSize());
    assertThat(nonAnomalies).isEqualTo(predict.getRowSize() - anomalies);
  }
}
