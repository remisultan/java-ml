package org.rsultan.dataframe;

import static java.util.stream.Collectors.toList;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Stream;

public record Row(List<?> values) implements Serializable {

  public Row(Object... values) {
    this(Stream.of(values).collect(toList()));
  }
}
