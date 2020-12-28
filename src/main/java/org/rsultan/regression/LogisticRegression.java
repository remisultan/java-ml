package org.rsultan.regression;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Dataframe;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

public class LogisticRegression extends AbstractLogisticRegression {

    private static final String YES = "Y";
    private static final String NO = "N";
    private String label;
    private INDArray YMatrix;

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

    public LogisticRegression setLabel(String label) {
        this.label = label;
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
    protected double computeLoss(INDArray prediction) {
        var h0 = YMatrix.mul(prediction);
        var h1 = YMatrix.neg().add(1).mul(prediction.neg().add(1));
        return h0.add(h1).mean().getDouble(0, 0);
    }

    @Override
    protected Function<String, String> formatPredictedLabel() {
        return label -> label.equals(YES) ? this.label : "Not " + this.label;
    }

    @Override
    public LogisticRegression train(Dataframe dataframe) {
        var df = dataframe
                .withColumn(this.label, responseVariableName, obj -> obj.toString().equals(this.label) ? YES : NO)
                .withColumn(INTERCEPT, () -> 1);

        X = df.toMatrix(predictorNames);
        XMean = X.mean(true, 1);
        X = X.div(XMean);
        Xt = X.transpose();

        labels = df.get(this.label).stream().distinct().sorted().map(Object::toString).collect(toList());

        YoneHot = df.oneHotEncode(this.label).select(YES, NO).toMatrix();
        Y = YoneHot.argMax(1).castTo(DataType.DOUBLE);
        W = Nd4j.ones(X.columns(), YoneHot.columns());

        YMatrix = Nd4j.create(Y.toDoubleVector(), Y.columns(), 1);

        this.run(X, Xt, Y, YoneHot);
        return this;
    }
}
