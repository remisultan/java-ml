package org.rsultan.dataframe.engine.mapper.impl.group;

import java.util.Locale;
import java.util.function.Predicate;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public record Aggregation(Object target, AggregationType type) {

  public String getColumnName() {
    return type.name().toLowerCase(Locale.ROOT) + "(" + target.toString() + ")";
  }
}
