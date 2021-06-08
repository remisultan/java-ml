package org.rsultan.core;

import java.io.Serializable;
import org.rsultan.dataframe.Dataframe;

public interface Trainable<T> extends Serializable {

  T train(Dataframe dataframe);

  Dataframe predict(Dataframe dataframe);

}
