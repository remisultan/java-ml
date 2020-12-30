package org.rsultan.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import java.util.ArrayList;
import java.util.stream.LongStream;

import static java.util.stream.LongStream.range;

public abstract class GradientDescentRegression extends AbstractRegression {

    static final String LOSS_COLUMN = "loss";
    static final String ACCURACY_COLUMN = "accuracy";

    protected final int numbersOfIterations;
    protected final double alpha;

    protected Dataframe history;
    private int lossAccuracyOffset;

    protected GradientDescentRegression(int numbersOfIterations, double alpha) {
        this.numbersOfIterations = numbersOfIterations;
        this.alpha = alpha;
        setLossAccuracyOffset(100);
    }

    protected void run(){
        var loss = new Column<>(LOSS_COLUMN, new ArrayList<Double>());
        var accuracy = new Column<>(ACCURACY_COLUMN, new ArrayList<Double>());

        range(0, this.numbersOfIterations)
                .map(idx -> {
                    var gradAlpha = computeGradient().mul(this.alpha);
                    W = W.sub(gradAlpha);
                    return idx;
                })
                .forEach(idx -> {
                    if(idx % getLossAccuracyOffset() == 0){
                        var prediction = computeNullHypothesis(X, W);
                        loss.values().add(computeLoss(prediction));
                        accuracy.values().add(computeAccuracy(X, W, Y));
                    }
                });

        this.history = Dataframes.create(loss, accuracy);
    }

    protected abstract INDArray computeGradient();

    protected double computeAccuracy(INDArray X, INDArray W, INDArray Y) {
        return LongStream.range(0, X.rows()).parallel()
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

    public int getLossAccuracyOffset() {
        return lossAccuracyOffset;
    }

    public void setLossAccuracyOffset(int lossAccuracyOffset) {
        this.lossAccuracyOffset = lossAccuracyOffset;
    }

}
