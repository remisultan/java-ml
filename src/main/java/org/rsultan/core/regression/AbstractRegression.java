package org.rsultan.core.regression;

import java.util.Arrays;
import java.util.stream.Stream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.ModelParameters;

public abstract class AbstractRegression
    extends ModelParameters<Regression>
    implements Regression {

  public static final String INTERCEPT = "Intercept";

  protected INDArray X;
  protected INDArray Xt;
  protected INDArray XMean;
  protected INDArray Y;
  protected INDArray W;

  public Regression setPredictorNames(String... names) {
    String[] strings = {INTERCEPT};
    String[] predictorNames = Stream.of(strings, names)
        .flatMap(Arrays::stream).distinct()
        .toArray(String[]::new);
    super.setPredictorNames(predictorNames);
    return this;
  }
}
