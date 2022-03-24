package org.rsultan.dataframe;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;

import java.io.Serializable;
import java.util.List;

public record Column<T>(String columnName, List<T> values) implements Serializable {
}
