package org.rsultan.dataframe;


import java.io.Serializable;
import java.util.List;

public record Row(List<?> values) implements Serializable {

  public Object get(int index) {
    return values.get(index);
  }

  public void remove(int index) {
    values.remove(index);
  }
}
