package org.rsultan.dataframe.transform.split;

import static java.util.Arrays.stream;

import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.TrainTestDataframe;

public record SplitDataframe(TrainTestDataframe dataframe) implements SplitTransform {

  @Override
  public TrainTestSplit split() {
    double splitValue = dataframe.getSplitValue();
    var columns = dataframe.getColumns();
    int splitBoundary = (int) (dataframe.getRowSize() * splitValue);
    var trainData = stream(columns).map(column ->
        new Column<>(column.columnName(), column.values().subList(0, splitBoundary))
    ).toArray(Column[]::new);
    var testData = stream(columns).map(column ->
        new Column<>(column.columnName(), column.values().subList(splitBoundary, dataframe.getRowSize()))
    ).toArray(Column[]::new);
    return new TrainTestSplit(Dataframes.create(trainData), Dataframes.create(testData));
  }

  public static record TrainTestSplit(Dataframe train, Dataframe test) {

  }
}
