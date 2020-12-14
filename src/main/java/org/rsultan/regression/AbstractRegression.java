package org.rsultan.regression;

import java.util.Arrays;
import java.util.stream.Stream;

public abstract class AbstractRegression implements Regression {

    public static final String INTERCEPT = "Intercept";
    protected String responseVariableName = "Y";
    protected String[] predictorNames = {};
    protected String predictionColumnName = "predictions";

    protected Regression setResponseVariableName(String name) {
        this.responseVariableName = name;
        return this;
    }

    protected Regression setPredictionColumnName(String name) {
        this.predictionColumnName = name;
        return this;
    }

    protected Regression setPredictorNames(String... names) {
        String[] strings = {INTERCEPT};
        this.predictorNames = Stream.of(strings, names).flatMap(Arrays::stream).distinct().toArray(String[]::new);
        return this;

    }
}
