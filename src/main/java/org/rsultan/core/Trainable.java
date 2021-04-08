package org.rsultan.core;

import org.rsultan.dataframe.Dataframe;

public interface Trainable<T> {

  T train(Dataframe dataframe);

  Dataframe predict(Dataframe dataframe);

}
