package org.rsultan.dataframe.transform;

import java.io.Serializable;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import org.rsultan.dataframe.Dataframe;

public interface FilterTransform extends Serializable {

  <SOURCE1> Dataframe filter(String columnName, Predicate<SOURCE1> predicate);

  <SOURCE1, SOURCE2> Dataframe filter(
      String sourceColumn1,
      String sourceColumn2,
      BiPredicate<SOURCE1, SOURCE2> predicate
  );
}
