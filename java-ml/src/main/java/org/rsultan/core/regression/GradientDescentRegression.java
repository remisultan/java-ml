package org.rsultan.core.regression;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.core.regularization.Regularization;

import java.util.ArrayList;

import static java.util.stream.LongStream.range;

public abstract class GradientDescentRegression extends AbstractRegression {

  private static final String LOSS_COLUMN = "loss";
  private static final String ACCURACY_COLUMN = "accuracy";
  protected final int numbersOfIterations;
  protected final double alpha;
  protected Regularization regularization;
  protected Dataframe history;
  private int lossAccuracyOffset;
  private double lambda = 0.001D;

  protected GradientDescentRegression(int numbersOfIterations, double alpha) {
    this.numbersOfIterations = numbersOfIterations;
    this.alpha = alpha;
    setLossAccuracyOffset(100);
    setRegularization(Regularization.NONE);
    setLambda(0.001);
  }

  protected GradientDescentRegression setRegularization(Regularization regularization) {
    this.regularization = regularization;
    return this;
  }

  public GradientDescentRegression setLossAccuracyOffset(int lossAccuracyOffset) {
    this.lossAccuracyOffset = lossAccuracyOffset;
    return this;
  }

  public GradientDescentRegression setLambda(double lambda) {
    this.lambda = lambda;
    return this;
  }

  protected void run() {
    List<List<?>> lossAccurracyOffsetRows = new ArrayList<>();
    var regularizer = regularization.getRegularizer(W, lambda);

    range(0, this.numbersOfIterations).forEach(idx -> {
      var gradAlpha = computeGradient().mul(this.alpha).add(regularizer.gradientRegularize());
      W.subi(gradAlpha);
      if (idx % lossAccuracyOffset == 0) {
        var prediction = computeNullHypothesis(X, W);
        double lossValue = computeLoss(prediction);
        List<Object> lossAccurracyRow = List.of(
            lossValue + regularizer.regularize(),
            computeAccuracy(X, W, Y)
        );
        lossAccurracyOffsetRows.add(lossAccurracyRow);
      }
    });

    this.history = Dataframes.create(new String[]{LOSS_COLUMN, ACCURACY_COLUMN},
        lossAccurracyOffsetRows);
  }

  protected abstract INDArray computeGradient();

  protected double computeAccuracy(INDArray X, INDArray W, INDArray Y) {
    return range(0, X.rows()).parallel()
        .map(idx -> {
          var xRow = Nd4j.create(X.getRow(idx).toDoubleVector(), 1, X.columns());
          xRow = xRow.div(XMean);
          var predictions = computeNullHypothesis(xRow, W);
          var predictedValue = predictions.argMax(1).getLong(0);
          return predictedValue == Y.getLong(idx) ? 1 : 0;
        }).mapToDouble(idx -> idx).average().orElse(0);
  }

  public Dataframe getHistory() {
    return history;
  }
}
