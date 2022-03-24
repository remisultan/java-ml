package org.rsultan.core;

import java.io.File;
import java.util.UUID;

public class ModelSerdeTestUtils {

  public static <T> T serdeTrainable(T trainable) {
    try {
      String pathFile = UUID.randomUUID() + ".gz";
      Models.write(pathFile, trainable);
      var model = Models.<T>read("./" + pathFile);
      new File(pathFile).delete();
      return model;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

}
