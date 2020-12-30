package org.rsultan.regression.impl;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.regression.GradientDescentRegression;

import java.util.List;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;
import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.log;


public class SoftmaxRegression extends LogisticRegression {

    public SoftmaxRegression(int numbersOfIterations, double alpha) {
        super(numbersOfIterations, alpha);
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
    public SoftmaxRegression train(Dataframe dataframe) {
        var dataframeIntercept = dataframe.withColumn(INTERCEPT, () -> 1);
        X = dataframeIntercept.toMatrix(predictorNames);
        XMean = X.mean(true ,1);
        X = X.div(XMean);
        Xt = X.transpose();
        this.labels = dataframe.get(responseVariableName).stream()
                .distinct().sorted()
                .map(Object::toString)
                .collect(toList());
        YoneHot = dataframe.oneHotEncode(responseVariableName).select(labels.toArray(String[]::new)).toMatrix();
        Y = YoneHot.argMax(1).castTo(DataType.DOUBLE);
        W = Nd4j.ones(X.columns(), YoneHot.columns());
        this.run();
        return this;
    }

    @Override
    public INDArray computeNullHypothesis(INDArray X, INDArray W) {
        return computeSoftmax(X.mmul(W));
    }

    private INDArray computeSoftmax(INDArray z) {
        var exp = exp(z.sub(Nd4j.max(z)));
        return exp.div(exp.sum(true, 1));
    }

    protected INDArray computeGradient() {
        var prediction = computeNullHypothesis(X, W);
        return Xt.div(X.rows()).mmul(prediction.sub(YoneHot));
    }

    @Override
    public double computeLoss(INDArray predictions) {
        var logLikelihood = log(predictions).mul(YoneHot).sum(true, 1).neg();
        return logLikelihood.mean().getDouble(0, 0);
    }

    @Override
    protected  Function<String, String> formatPredictedLabel() {
        return String::valueOf;
    }
}
