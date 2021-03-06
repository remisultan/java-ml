package org.rsultan.core.dimred;

import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.nd4j.common.util.ArrayUtil.argsort;
import static org.nd4j.linalg.eigen.Eigen.symmetricGeneralizedEigenvalues;

import java.util.List;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.Trainable;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.utils.Matrices;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PrincipalComponentAnalysis implements Trainable<PrincipalComponentAnalysis> {

  private static final Logger LOG = LoggerFactory.getLogger(PrincipalComponentAnalysis.class);

  private final int numberOfComponent;
  private String responseVariable = "y";
  private INDArray Xmean;
  private INDArray eighenVectors;
  private INDArray predictions;
  private List<String> responseVariableData;

  public PrincipalComponentAnalysis(int numberOfComponents) {
    this.numberOfComponent = numberOfComponents;
  }

  @Override
  public PrincipalComponentAnalysis train(Dataframe dataframe) {
    var X = dataframe.mapWithout(responseVariable).toMatrix();
    this.responseVariableData = dataframe.get(responseVariable);
    int components = Math.min(numberOfComponent, X.columns());
    Xmean = X.mean(0);
    X = X.sub(Xmean);
    LOG.info("computing covariance matrix");
    LOG.info("computing eighenvectors");
    eighenVectors = Matrices.covariance(X);
    var eighenValuesArgSort = argsort(
        symmetricGeneralizedEigenvalues(eighenVectors, true).toIntVector(), false
    );
    eighenVectors = eighenVectors
        .getColumns(eighenValuesArgSort)
        .getColumns(range(0, components).toArray());
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var Xpredict = dataframe.mapWithout(responseVariable).toMatrix().sub(Xmean);
    LOG.info("computing predictions");
    predictions = eighenVectors.transpose().mmul(Xpredict.transpose()).transpose();
    List<Column<?>> columns = range(0, predictions.columns())
        .mapToObj(colIdx -> new Column<>("c" + colIdx, range(0, predictions.rows())
            .mapToObj(rowIdx -> predictions.getDouble(rowIdx, colIdx))
            .collect(toList()))
        ).collect(toList());
    columns.add(new Column<>(responseVariable, responseVariableData));
    return Dataframes.create(columns.toArray(Column[]::new));
  }

  public Dataframe reconstruct() {
    LOG.info("reconstructing original matrix");
    var XreBuilt = predictions.mmul(eighenVectors.transpose()).add(Xmean);
    List<Column<?>> columns = range(0, XreBuilt.columns())
        .mapToObj(colIdx -> new Column<>("c" + colIdx, range(0, XreBuilt.rows())
            .mapToObj(rowIdx -> XreBuilt.getDouble(rowIdx, colIdx))
            .collect(toList()))
        ).collect(toList());
    columns.add(new Column<>(responseVariable, responseVariableData));
    return Dataframes.create(columns.toArray(Column[]::new));
  }

  public PrincipalComponentAnalysis setResponseVariable(String responseVariable) {
    this.responseVariable = responseVariable;
    return this;
  }
}
