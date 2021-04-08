package org.rsultan.dataframe.transform.filter;

import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;

import java.util.List;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import org.rsultan.dataframe.Dataframe;

public interface FilterTransform {

  <SOURCE1> Dataframe filter(String columnName, Predicate<SOURCE1> predicate);

  <SOURCE1, SOURCE2> Dataframe filter(
      String sourceColumn1,
      String sourceColumn2,
      BiPredicate<SOURCE1, SOURCE2> predicate
  );

}
