package org.rsultan.dataframe.engine.filter;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import org.rsultan.dataframe.Row;

public class RowPredicate<T> extends FilterDataProcessor {

  private final Object columnName;
  private final Predicate<T> predicate;

  private AtomicInteger columnIndex;

  public RowPredicate(Object columnName, Predicate<T> predicate) {
    this.columnName = columnName;
    this.predicate = predicate;
  }

  @Override
  protected boolean filter(Row row) {
    return predicate.test((T) row.values().get(columnIndex.get()));
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndex = getColumnIndex(header, columnName);
    super.setHeader(header);
  }
}
