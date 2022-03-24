package org.rsultan.dataframe.engine.mapper.impl;

import static java.util.stream.Collectors.toList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.mapper.RowMapperDataProcessor;

public class SelectColumnMapper extends RowMapperDataProcessor {

  private final List<Object> columns;
  private List<Integer> columnIndices;

  public SelectColumnMapper(Object... columns) {
    this(Arrays.asList(columns));
  }

  public SelectColumnMapper(List<Object> columns) {
    super();
    this.columns = columns;
  }

  @Override
  public Row map(Row row) {
    if(columnIndices == null) {
      return row;
    }
    var newValues = new ArrayList<>(columnIndices.size());
    for (Integer columnIdx : columnIndices) {
      newValues.add(row.get(columnIdx));
    }
    return new Row(newValues);
  }

  @Override
  public void start(ExecutorService executorService) {
    super.start(executorService);
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndices = columns.stream().filter(header::contains).map(header::indexOf).collect(toList());
    super.setHeader(columns);
  }
}
