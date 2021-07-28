package org.rsultan.dataframe.transform.shuffle;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

import java.util.Collections;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.TrainTestDataframe;

public record ShuffleDataframe(TrainTestDataframe dataframe) implements
    ShuffleTransform<TrainTestDataframe> {

  @Override
  public TrainTestDataframe shuffle() {
    var columns = dataframe.getColumns();
    var rows = range(0, dataframe.getRowSize()).parallel()
        .mapToObj(idx -> new Row(
            stream(columns).map(column -> column.values().get(idx)).collect(toList())))
        .collect(toList());
    Collections.shuffle(rows);
    var columnNames = stream(columns).map(Column::columnName).toArray(String[]::new);
    return Dataframes.trainTest(columnNames, rows.toArray(Row[]::new));
  }
}
