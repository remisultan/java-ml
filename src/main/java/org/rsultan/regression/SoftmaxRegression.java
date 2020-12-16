package org.rsultan.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.log;


public class SoftmaxRegression extends AbstractRegression {

    public static final String LOSS_COLUMN = "loss";
    public static final String ACCURACY_COLUMN = "accuracy";
    private static final Logger LOG = LoggerFactory.getLogger(SoftmaxRegression.class);
    private final int numbersOfIterations;
    private final double alpha;
    private INDArray W;
    private List<String> labels;
    private Dataframe history;

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
    public SoftmaxRegression train(Dataframe dataframe) {
        var dataframeIntercept = dataframe.withColumn(INTERCEPT, () -> 1);
        var X = dataframeIntercept.toMatrix(predictorNames);
        var Xt = X.transpose();
        this.labels = dataframe.get(responseVariableName).stream().distinct().sorted().map(Object::toString).collect(toList());
        var YoneHot = dataframe.oneHotEncode(responseVariableName)
                .withoutColumn(responseVariableName)
                .withoutColumn(predictorNames).toMatrix();
        W = Nd4j.ones(X.columns(), YoneHot.columns());

        var loss = new Column<>(LOSS_COLUMN, new ArrayList<Double>());
        var accuracy = new Column<>(ACCURACY_COLUMN, new ArrayList<Double>());

        range(0, this.numbersOfIterations).boxed()
                .map(idx -> this.computeGradient(W, X, Xt, YoneHot))
                .map(gradient -> {
                    System.out.println(this.computeNullHypothesis(X, W));
                    var gradAlpha = gradient.mul(this.alpha);
                    W.subi(gradAlpha);
                    return W;
                })
                .map(W -> computeNullHypothesis(X, W))
                .map(prediction -> Map.entry(computeLoss(prediction, YoneHot), computeAccuracy(X)))
                .forEach(entry -> {
                    loss.values().add(entry.getKey());
                    accuracy.values().add(entry.getValue());
                    LOG.info("Loss: {}", entry.getKey());
                    LOG.info("Accuracy: {}", entry.getValue());
                });

        this.history = Dataframes.create(loss, accuracy);
        return this;
    }

    private double computeAccuracy(INDArray x) {
        return IntStream.range(0, labels.size())
                .filter(idx -> {
                    var xRow = Nd4j.create(x.getRow(idx).toDoubleVector(), 1, x.columns());
                    var predictions = computeNullHypothesis(xRow, W);
                    int predictedValue = Nd4j.argMax(predictions, 1).getInt(0);
                    return labels.get(predictedValue).equals(labels.get(idx));
                }).mapToDouble(idx -> 1D)
                .sum() / labels.size();
    }

    private double computeLoss(INDArray predictions, INDArray oneHot) {
        return predictions.sub(oneHot).mmul(log(predictions).transpose()).sum().getDouble(0, 0);
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
        var exp = exp(z.sub(Nd4j.max(z)));
        return exp.div(exp.sum(true, 1));
    }

    @Override
    public Dataframe predict(Dataframe dataframe) {
        var dataframeIntercept = dataframe.withColumn(INTERCEPT, () -> 1);
        var X = dataframeIntercept.toMatrix(predictorNames);
        var predictions = computeNullHypothesis(X, W);
        var predictionList = range(0, predictions.rows()).boxed()
                .map(predictions::getRow)
                .map(row -> Nd4j.argMax(row).getInt(0))
                .map(labels::get)
                .collect(toList());
        var columns = new Column<>(predictionColumnName, predictionList);
        return dataframe.addColumn(columns);
    }

    public Dataframe getHistory() {
        return history;
    }
}
