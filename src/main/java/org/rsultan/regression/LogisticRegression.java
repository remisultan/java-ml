package org.rsultan.regression;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframe;

import static java.util.stream.Collectors.toList;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

public class LogisticRegression extends GradientDescentRegression {

    public static final String YONE_HOT = "YoneHot";

    public LogisticRegression(int numbersOfIterations, double alpha) {
        super(numbersOfIterations, alpha);
    }

    @Override
    public LogisticRegression setResponseVariableName(String name) {
        super.setResponseVariableName(name);
        return this;
    }

    @Override
    public LogisticRegression setPredictionColumnName(String name) {
        super.setPredictionColumnName(name);
        return this;
    }

    @Override
    public LogisticRegression setPredictorNames(String... names) {
        super.setPredictorNames(names);
        return this;
    }

    @Override
    protected INDArray computeNullHypothesis(INDArray X, INDArray W) {
        return computeSigmoid(X.mmul(W));
    }

    private INDArray computeSigmoid(INDArray z) {
        return sigmoid(z);
    }

    @Override
    protected INDArray computeGradient(INDArray X, INDArray Xt, INDArray W, INDArray labels) {
        var prediction = computeNullHypothesis(X, W);
        return Xt.div(X.rows()).mmul(prediction.sub(labels));
    }

    @Override
    protected double computeLoss(INDArray prediction, INDArray trueLabels) {
        var logLikelihood = log(prediction).mul(trueLabels).sum(true, 1).neg();
        return logLikelihood.mean().getDouble(0, 0);
    }

    @Override
    public LogisticRegression train(Dataframe dataframe) {
        super.train(dataframe);
        return this;
    }
}
