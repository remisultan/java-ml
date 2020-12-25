package org.rsultan.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;

import java.util.ArrayList;
import java.util.Map;
import java.util.stream.LongStream;

import static java.util.stream.LongStream.range;

public abstract class GradientDescentRegression extends AbstractRegression {

    private static final String LOSS_COLUMN = "loss";
    private static final String ACCURACY_COLUMN = "accuracy";

    protected final int numbersOfIterations;
    protected final double alpha;
    protected INDArray W;
    private Dataframe history;

    protected GradientDescentRegression(int numbersOfIterations, double alpha) {
        this.numbersOfIterations = numbersOfIterations;
        this.alpha = alpha;
    }

    protected abstract INDArray computeNullHypothesis(INDArray X, INDArray W);

    protected abstract INDArray computeGradient(INDArray X, INDArray Xt, INDArray W, INDArray oneHotEncodedLabels);

    protected abstract double computeLoss(INDArray prediction, INDArray trueLabels);

    protected void run(INDArray X, INDArray Xt, INDArray Y, INDArray YoneHot){
        W = Nd4j.ones(X.columns(), YoneHot.columns());

        var loss = new Column<>(LOSS_COLUMN, new ArrayList<Double>());
        var accuracy = new Column<>(ACCURACY_COLUMN, new ArrayList<Double>());

        range(0, this.numbersOfIterations).boxed()
                .map(idx -> this.computeGradient(X, Xt, W, YoneHot))
                .map(gradient -> {
                    var gradAlpha = gradient.mul(this.alpha);
                    W = W.sub(gradAlpha);
                    return W;
                })
                .map(W -> Map.entry(computeLoss(computeNullHypothesis(X, W), YoneHot), computeAccuracy(X, W, Y)))
                .forEach(entry -> {
                    loss.values().add(entry.getKey());
                    accuracy.values().add(entry.getValue());
                });

        this.history = Dataframes.create(loss, accuracy);
    };

    protected double computeAccuracy(INDArray X, INDArray W, INDArray Y) {
        return LongStream.range(0, X.rows()).parallel()
                .map(idx -> {
                    var xRow = Nd4j.create(X.getRow(idx).toDoubleVector(), 1, X.columns());
                    var predictions = computeNullHypothesis(xRow, W);
                    var predictedValue = predictions.argMax(1).getLong(0);
                    return predictedValue == Y.getLong(idx) ? 1 : 0;
                }).mapToDouble(idx -> idx).average().orElse(0);
    }

    public Dataframe getHistory() {
        return history;
    }
}
