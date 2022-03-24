package org.rsultan.core.dimred;

import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.nd4j.common.util.ArrayUtil.argsort;
import static org.nd4j.linalg.eigen.Eigen.symmetricGeneralizedEigenvalues;
import static org.nd4j.linalg.ops.transforms.Transforms.normalizeZeroMeanAndUnitVariance;

import java.util.List;
import java.util.stream.Stream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.RawTrainable;
import org.rsultan.core.Trainable;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.utils.Matrices;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PrincipalComponentAnalysis implements
    Trainable<PrincipalComponentAnalysis>, RawTrainable<PrincipalComponentAnalysis> {

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
    var X = normalizeZeroMeanAndUnitVariance(dataframe.copy().mapWithout(responseVariable).toMatrix());
    this.responseVariableData = dataframe.getColumn(responseVariable);
    return this.train(X);
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var Xpredict = dataframe.copy().mapWithout(responseVariable).toMatrix();
    LOG.info("computing predictions");
    this.predict(Xpredict);
    var columns = range(0, predictions.columns()).mapToObj(colIdx -> "c" + colIdx)
        .toArray(String[]::new);
    List<List<?>> rows = range(0, predictions.rows()).mapToObj(rowId -> range(0, predictions.columns())
        .mapToObj(colIdx -> predictions.getDouble(rowId, colIdx))
        .collect(toList())).collect(toList());

    return Dataframes.create(columns, rows).addColumn(responseVariable, responseVariableData);
  }

  @Override
  public PrincipalComponentAnalysis train(INDArray X) {
    int components = Math.min(numberOfComponent, X.columns());
    Xmean = X.mean(0);
    X = X.sub(Xmean);
    LOG.info("computing covariance matrix");
    eighenVectors = Matrices.covariance(X);
    LOG.info("computing eighenvectors");
    var eighenValuesArgSort = argsort(
        symmetricGeneralizedEigenvalues(eighenVectors, true).toIntVector(), false
    );
    eighenVectors = eighenVectors
        .getColumns(eighenValuesArgSort)
        .getColumns(range(0, components).toArray());
    LOG.info("eighenvectors computed");
    return this;
  }

  public Dataframe reconstruct() {
    LOG.info("trying to reconstruct original matrix");
    var XreBuilt = rawReconstruct();
    var columns = range(0, XreBuilt.columns()).mapToObj(colIdx -> "c" + colIdx)
        .toArray(String[]::new);
    List<List<?>> rows = range(0, XreBuilt.rows()).mapToObj(rowIdx -> range(0, XreBuilt.columns())
        .mapToObj(colIdx -> XreBuilt.getDouble(rowIdx, colIdx))
        .collect(toList())).collect(toList());

    return Dataframes.create(columns, rows).addColumn(responseVariable, responseVariableData);
  }

  public INDArray rawReconstruct() {
    return predictions.mmul(eighenVectors.transpose()).add(Xmean);
  }

  @Override
  public INDArray predict(INDArray matrix) {
    predictions = eighenVectors.transpose().mmul(matrix.sub(Xmean).transpose()).transpose();
    return predictions;
  }

  public PrincipalComponentAnalysis setResponseVariable(String responseVariable) {
    this.responseVariable = responseVariable;
    return this;
  }

}
