package org.rsultan.dataframe;

import java.util.List;

public record Column<T>(String columnName, List<T> values){}
