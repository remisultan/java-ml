package org.rsultan.dataframe.engine.mapper.impl;

import org.rsultan.dataframe.Row;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.rsultan.dataframe.engine.mapper.RowMapperDataProcessor;

import static java.lang.String.format;
import static java.util.Optional.ofNullable;
import static java.util.stream.Collectors.toList;

public class RemoveColumnMapper extends RowMapperDataProcessor {

  private final Object column;
  private AtomicInteger columnIndex;

  public RemoveColumnMapper(Object column) {
    super();
    this.column = column;
  }

  @Override
  public Row map(Row row) {
    if (columnIndex.get() != -1) {
      row.remove(columnIndex.get());
    }
    return new Row(row.values());
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndex = ofNullable(columnIndex).orElse(new AtomicInteger(header.indexOf(column)));
    if (columnIndex.get() != -1) {
      var columns = header.stream().filter(o -> columnIndex.get() != header.indexOf(o))
          .collect(toList());
      super.setHeader(columns);
    }
  }
}
