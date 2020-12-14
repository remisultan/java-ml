package org.rsultan.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;

import java.util.ArrayList;
import java.util.Map;

import static java.util.stream.LongStream.range;
import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.log;


public class SoftmaxRegression extends AbstractRegression {

    public static final String LOSS_COLUMN = "loss";
    public static final String ACCURACY_COLUMN = "accuracy";
    private final int numbersOfIterations;
    private final double alpha;
    private INDArray W;

    public SoftmaxRegression(
            int numbersOfIterations,
            double alpha) {
        this.numbersOfIterations = numbersOfIterations;
        this.alpha = alpha;
    }

    @Override
    public SoftmaxRegression setResponseVariableName(String name) {
        super.setResponseVariableName(name);
        return this;
    }

    @Override
    public SoftmaxRegression setPredictionColumnName(String name) {
        super.setPredictionColumnName(name);
        return this;
    }

    @Override
    public SoftmaxRegression setPredictorNames(String... names) {
        super.setPredictorNames(names);
        return this;
    }

    @Override
    public Regression train(Dataframe dataframe) {
        var dataframeIntercept = dataframe.withColumn(INTERCEPT, () -> 1);
        var X = dataframeIntercept.toMatrix(predictorNames);
        var Xt = X.transpose();
        var Y = dataframe.toMatrix(responseVariableName);
        var YoneHot = dataframe.oneHotEncode(responseVariableName)
                .withoutColumn(responseVariableName)
                .withoutColumn(predictorNames).toMatrix();
        W = Nd4j.ones(X.columns(), YoneHot.columns());

        var loss = new Column<>(LOSS_COLUMN,  new ArrayList<Double>());
        var accuracy = new Column<>(ACCURACY_COLUMN,  new ArrayList<Double>());

        range(0, this.numbersOfIterations).boxed()
                .map(idx -> this.computeGradient(W, X, Xt, YoneHot))
                .map(gradient -> gradient.mul(this.alpha * -1))
                .map(gradient -> {
                    W.subi(gradient);
                    return W;
                })
                .map(W -> computeNullHypothesis(X, W))
                .map(prediction -> Map.entry(computeLoss(prediction, Y), computeAccuracy(X, Y)))
                .forEach(entry -> {
                    loss.values().add(entry.getKey());
                    accuracy.values().add(entry.getValue());
                });

        Dataframes.create(loss, accuracy).show(this.numbersOfIterations);
        return this;
    }

    private double computeAccuracy(INDArray x, INDArray y) {
        return range(0, y.rows())
                .filter(idx -> {
                    var xRow = Nd4j.create(x.getRow(idx).toDoubleVector(), 1, x.columns());
                    var predictions = computeNullHypothesis(xRow, W);
                    double predictedValue = Nd4j.argMax(predictions).getDouble(0);
                    return predictedValue == y.getDouble(idx, 0);
                }).mapToDouble(idx -> 1D)
                .sum() / (double) y.rows();
    }

    private double computeLoss(INDArray predictions, INDArray y) {
        return log(predictions).mul(-1).sum().div(y.rows()).getDouble(0, 0);
    }

    private INDArray computeGradient(INDArray W,
                                     INDArray X,
                                     INDArray Xt,
                                     INDArray yOneHot) {
        var prediction = computeNullHypothesis(X, W);
        return Xt.div(X.rows()).mmul(prediction.sub(yOneHot));
    }

    private INDArray computeNullHypothesis(INDArray X, INDArray W) {
        return computeSoftmax(X.mmul(W));
    }

    private INDArray computeSoftmax(INDArray z) {
        var normalizedZ = z.sub(z.max());
        var exp = exp(normalizedZ);
        return exp.div(exp.sum());
    }

    @Override
    public Dataframe predict(Dataframe dataframe) {
        return null;
    }
}
