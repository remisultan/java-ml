package org.rsultan.dataframe;

import java.util.List;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;

public record Column<T>(String columnName, List<T> values){

    public Column(String columnName, T... values) {
        this(columnName, Stream.of(values).collect(toList()));
    }
}
