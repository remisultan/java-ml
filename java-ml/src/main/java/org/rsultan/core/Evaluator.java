package org.rsultan.core;

import org.rsultan.core.Trainable;
import org.rsultan.dataframe.TrainTestDataframe;

public interface Evaluator<V, T extends Trainable<T>> {

  V evaluate(T trainable, TrainTestDataframe dataframe);

}
