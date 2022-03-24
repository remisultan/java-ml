package org.rsultan.dataframe.transform;

import java.io.Serializable;
import org.rsultan.dataframe.Dataframe;

public interface ShuffleTransform extends Serializable {

  Dataframe shuffle();

}
