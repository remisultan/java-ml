package org.rsultan.core.serde;

import java.io.IOException;

public interface Deserialiser<S, T> {

  T deserialise(S source) throws IOException, ClassNotFoundException;

}
