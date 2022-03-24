package org.rsultan.dataframe.engine.filter;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiPredicate;
import org.rsultan.dataframe.Row;

public class RowBiPredicate<T, U> extends FilterDataProcessor {

  private final Object columnName;
  private final Object columnName2;
  private final BiPredicate<T, U> predicate;

  private AtomicInteger columnIndex;
  private AtomicInteger columnIndex2;

  public RowBiPredicate(Object columnName, Object columnName2, BiPredicate<T, U> predicate) {
    this.columnName = columnName;
    this.predicate = predicate;
    this.columnName2 = columnName2;
  }

  @Override
  protected boolean filter(Row row) {
    return predicate.test((T) row.get(columnIndex.get()), (U) row.get(columnIndex2.get()));
  }

  @Override
  public void setHeader(List<Object> header) {
    columnIndex = getColumnIndex(header, columnName);
    columnIndex2 = getColumnIndex(header, columnName2);
    super.setHeader(header);
  }
}
