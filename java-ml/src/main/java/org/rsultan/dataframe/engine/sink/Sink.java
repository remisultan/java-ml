package org.rsultan.dataframe.engine.sink;

import org.rsultan.dataframe.Row;

public interface Sink {

  void consume(Row row);
}
